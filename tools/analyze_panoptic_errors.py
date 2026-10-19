#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CLASS_NAMES = {0: "ship", 1: "obstacle"}
ALL_CLASS_NAMES = {0: "ship", 1: "obstacle", 2: "seasurface", 3: "backdrop"}
THING_CLASSES = tuple(sorted(CLASS_NAMES))
SIZE_NAMES = ("small", "medium", "large")


def select_static_tiles(
    height,
    width,
    tile_size,
    overlap,
    budget,
    mode,
    *,
    seed=0,
    image_index=0,
):
    from tile_geometry import sliding_tiles

    if budget is None or budget <= 0:
        raise ValueError("static tile budget must be a positive integer")
    if mode not in {"center", "random", "horizon-row"}:
        raise ValueError(f"unsupported static tile selector: {mode}")

    windows = sliding_tiles(height, width, tile_size, overlap)
    count = min(int(budget), len(windows))
    if mode == "horizon-row":
        y = 896 if (height, width)==(2800,2800) else 0 if (height,width)==(1600,2800) else None
        if y is None or list(tile_size)!=[1120,1120] or overlap!=0.2 or budget!=3:
            raise ValueError("horizon-row protocol requires supported frame sizes, 1120 windows, overlap .2, K3")
        return [i for i,w in enumerate(windows) if w[0]==y]
    if mode == "random":
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(image_index)]))
        return rng.choice(len(windows), size=count, replace=False).tolist()

    center_y, center_x = height / 2.0, width / 2.0
    ranked = sorted(
        range(len(windows)),
        key=lambda index: (
            ((windows[index][0] + windows[index][2]) / 2.0 - center_y) ** 2
            + ((windows[index][1] + windows[index][3]) / 2.0 - center_x) ** 2,
            index,
        ),
    )
    return ranked[:count]


def size_bin(area: int) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def small_target_tile_coverage(
    target,
    height,
    width,
    tile_size,
    overlap,
    thing_classes=THING_CLASSES,
    max_area=32**2,
):
    from tools.tiled_panoptic_infer import sliding_tiles

    if max_area <= 0:
        raise ValueError("oracle max area must be positive")

    windows = sliding_tiles(height, width, tile_size, overlap)
    masks = target["masks"].as_subclass(torch.Tensor).bool()
    labels = torch.as_tensor(target["labels"])
    is_crowd = torch.as_tensor(target["is_crowd"]).bool()
    areas = masks.flatten(1).sum(1)
    eligible = (
        torch.isin(labels, torch.as_tensor(thing_classes, dtype=labels.dtype))
        & ~is_crowd
        & (areas > 0)
        & (areas < max_area)
    )
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    num_targets = len(eligible_indices)

    coverage = np.zeros((num_targets, len(windows)), dtype=np.float64)
    for row, mask_index in enumerate(eligible_indices):
        ys, xs = torch.nonzero(masks[mask_index], as_tuple=True)
        area = float(areas[mask_index])
        for column, (y0, x0, y1, x1) in enumerate(windows):
            visible = ((ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)).sum()
            coverage[row, column] = float(visible) / area
    return windows, coverage


def selected_small_tile_diagnostics(
    target,
    height,
    width,
    tile_size,
    overlap,
    selected,
    thing_classes=THING_CLASSES,
    max_area=32**2,
):
    windows, coverage = small_target_tile_coverage(
        target, height, width, tile_size, overlap, thing_classes, max_area
    )
    selected = list(selected)
    best_coverage = (
        coverage[:, selected].max(axis=1)
        if selected and len(coverage)
        else np.zeros(len(coverage), dtype=np.float64)
    )
    return {
        "images": 1,
        "available_tiles": len(windows),
        "selected_tiles": len(selected),
        "small_targets": len(coverage),
        "coverage_sum": float(best_coverage.sum()),
        "covered_50": int((best_coverage >= 0.5).sum()),
        "covered_99": int((best_coverage >= 0.99).sum()),
        "images_without_selected_tiles": int(not selected),
    }


def select_oracle_small_tiles(
    target,
    height,
    width,
    tile_size,
    overlap,
    budget,
    thing_classes=THING_CLASSES,
    max_area=32**2,
):
    if budget is None or budget <= 0:
        raise ValueError("oracle tile budget must be a positive integer")
    windows, coverage = small_target_tile_coverage(
        target, height, width, tile_size, overlap, thing_classes, max_area
    )

    selected = []
    best_coverage = np.zeros(len(coverage), dtype=np.float64)
    for _ in range(min(int(budget), len(windows))):
        gains = np.maximum(best_coverage[:, None], coverage).sum(axis=0)
        gains -= best_coverage.sum()
        if selected:
            gains[selected] = -np.inf
        best_index = int(np.argmax(gains)) if len(gains) else -1
        if best_index < 0 or gains[best_index] <= 1e-12:
            break
        selected.append(best_index)
        best_coverage = np.maximum(best_coverage, coverage[:, best_index])

    diagnostics = selected_small_tile_diagnostics(
        target, height, width, tile_size, overlap, selected, thing_classes, max_area
    )
    return selected, diagnostics


def instances_from_map(pixels: torch.Tensor, thing_classes=(0, 1), is_crowd=None):
    result = []
    for class_id in thing_classes:
        class_pixels = pixels[..., 0] == class_id
        for instance_id in torch.unique(pixels[..., 1][class_pixels]).tolist():
            if instance_id < 0:
                continue
            if is_crowd is not None and bool(is_crowd[instance_id]):
                continue
            mask = class_pixels & (pixels[..., 1] == instance_id)
            area = int(mask.sum())
            if area:
                result.append({"class_id": class_id, "mask": mask, "area": area})
    return result


def pair_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    intersection = int((a & b).sum())
    union = int(a.sum()) + int(b.sum()) - intersection
    return intersection / union if union else 0.0


def update_stats(
    stats,
    gt_instances,
    pred_instances,
    instance_records=None,
    record_context=None,
    ignored_regions=None,
):
    pairs = []
    for gi, gt in enumerate(gt_instances):
        for pi, pred in enumerate(pred_instances):
            if gt["class_id"] == pred["class_id"]:
                iou = pair_iou(gt["mask"], pred["mask"])
                if iou > 0.5:
                    pairs.append((iou, gi, pi))

    matched_gt, matched_pred = set(), set()
    for iou, gi, pi in sorted(pairs, reverse=True):
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        gt = gt_instances[gi]
        key = (gt["class_id"], size_bin(gt["area"]))
        stats[key]["tp"] += 1
        stats[key]["iou_sum"] += iou

    if instance_records is not None:
        context = dict(record_context or {})
        for gi, gt in enumerate(gt_instances):
            instance_records.append(
                {
                    **context,
                    "class_id": int(gt["class_id"]),
                    "class_name": CLASS_NAMES[int(gt["class_id"])],
                    "area": int(gt["area"]),
                    "matched": gi in matched_gt,
                }
            )

    for gi, gt in enumerate(gt_instances):
        if gi in matched_gt:
            continue
        key = (gt["class_id"], size_bin(gt["area"]))
        stats[key]["fn"] += 1
        same = [
            pair_iou(gt["mask"], pred["mask"])
            for pred in pred_instances
            if pred["class_id"] == gt["class_id"]
        ]
        other = [
            pair_iou(gt["mask"], pred["mask"])
            for pred in pred_instances
            if pred["class_id"] != gt["class_id"]
        ]
        best_same = max(same, default=0.0)
        best_other = max(other, default=0.0)
        stats[key]["best_iou_sum"] += best_same
        if best_other > 0.5:
            stats[key]["wrong_class"] += 1
        elif best_same >= 0.1:
            stats[key]["poor_mask"] += 1
        else:
            stats[key]["no_overlap"] += 1

    for pi, pred in enumerate(pred_instances):
        if pi not in matched_pred:
            ignored = (ignored_regions or {}).get(pred["class_id"])
            if ignored is not None and float((pred["mask"] & ignored).sum()) / pred["area"] > 0.5:
                continue
            stats[(pred["class_id"], size_bin(pred["area"]))]["fp"] += 1


def stats_rows(stats, sequence=None):
    rows = []
    for class_id in sorted(CLASS_NAMES):
        for size in SIZE_NAMES:
            values = stats[(class_id, size)]
            tp, fn, fp = (int(values[k]) for k in ("tp", "fn", "fp"))
            row = {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "size": size,
                "gt": tp + fn,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "mean_matched_iou": values["iou_sum"] / tp if tp else 0.0,
                "fn_no_overlap": int(values["no_overlap"]),
                "fn_poor_mask": int(values["poor_mask"]),
                "fn_wrong_class": int(values["wrong_class"]),
                "fn_mean_best_same_class_iou": (
                    values["best_iou_sum"] / fn if fn else 0.0
                ),
            }
            if sequence is not None:
                row = {"sequence": sequence, **row}
            rows.append(row)
    return rows


def compare_rows(candidate_rows, baseline_rows):
    baseline = {
        (int(row["class_id"]), row["size"]): row for row in baseline_rows
    }
    compared = []
    for row in candidate_rows:
        key = (int(row["class_id"]), row["size"])
        if key not in baseline:
            continue
        base = baseline[key]
        compared.append(
            {
                "class_id": key[0],
                "class_name": row["class_name"],
                "size": key[1],
                "candidate_gt": int(row["gt"]),
                "baseline_gt": int(base["gt"]),
                "delta_tp": int(row["tp"]) - int(base["tp"]),
                "delta_fn": int(row["fn"]) - int(base["fn"]),
                "delta_fp": int(row["fp"]) - int(base["fp"]),
                "delta_recall": float(row["recall"]) - float(base["recall"]),
                "delta_mean_matched_iou": (
                    float(row["mean_matched_iou"])
                    - float(base["mean_matched_iou"])
                ),
            }
        )
    return compared


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["class_id", "class_name", "size"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def init_pq_metric(model, device="cpu"):
    metric = model.metrics[-1]
    metric.reset()
    metric.to(device)
    return metric


def pq_summary(metric):
    result = metric.compute()[:-1].cpu()
    num_things = len(metric.things)
    ordered_ids = [
        cat_id
        for cat_id, _ in sorted(
            metric.cat_id_to_continuous_id.items(), key=lambda kv: kv[1]
        )
    ][: len(result)]

    def group(sl):
        block = result[sl]
        if not len(block):
            return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "num_classes": 0}
        return {
            "pq": float(block[:, 0].mean()),
            "sq": float(block[:, 1].mean()),
            "rq": float(block[:, 2].mean()),
            "num_classes": int(len(block)),
        }

    summary = {
        "all": group(slice(None)),
        "things": group(slice(0, num_things)),
        "stuff": group(slice(num_things, None)),
        "per_class": [
            {
                "class_id": int(cat_id),
                "class_name": ALL_CLASS_NAMES.get(int(cat_id), str(int(cat_id))),
                "pq": float(result[i, 0]),
                "sq": float(result[i, 1]),
                "rq": float(result[i, 2]),
                "tp": int(metric.true_positives[i]),
                "fp": int(metric.false_positives[i]),
                "fn": int(metric.false_negatives[i]),
            }
            for i, cat_id in enumerate(ordered_ids)
        ],
    }
    return summary


def compare_pq(candidate, baseline):
    if not candidate or not baseline:
        return None
    deltas = {}
    for group in ("all", "things", "stuff"):
        if group in candidate and group in baseline:
            deltas[group] = {
                key: float(candidate[group][key]) - float(baseline[group][key])
                for key in ("pq", "sq", "rq")
            }
    base_by_class = {
        int(row["class_id"]): row for row in baseline.get("per_class", [])
    }
    deltas["per_class"] = [
        {
            "class_id": int(row["class_id"]),
            "class_name": row["class_name"],
            **{
                f"delta_{key}": float(row[key])
                - float(base_by_class[int(row["class_id"])][key])
                for key in ("pq", "sq", "rq")
            },
        }
        for row in candidate.get("per_class", [])
        if int(row["class_id"]) in base_by_class
    ]
    return deltas


def panoptic_ids_to_pixels(panoptic_ids, segments_info, num_classes=None):
    ids = torch.as_tensor(panoptic_ids, dtype=torch.long)
    pixels = torch.full((*ids.shape, 2), -1, dtype=torch.long)
    if num_classes is not None:
        pixels[..., 0] = int(num_classes)
    for instance_id, segment in enumerate(segments_info):
        mask = ids == int(segment["id"])
        if mask.any():
            pixels[..., 0][mask] = int(segment["category_id"])
            pixels[..., 1][mask] = instance_id
    return pixels


@torch.no_grad()
def predict(
    model,
    img,
    device,
    inference_mode="standard",
    tiled_kwargs=None,
    tile_model=None,
):
    if inference_mode != "standard":
        from tools.tiled_panoptic_infer import infer_tiled_panoptic

        panoptic_ids, segments_info = infer_tiled_panoptic(
            model,
            img,
            device,
            mode=inference_mode,
            tile_model=tile_model,
            **(tiled_kwargs or {}),
        )
        conversion_started = time.perf_counter()
        pred = panoptic_ids_to_pixels(
            panoptic_ids, segments_info, num_classes=model.num_classes
        )
        timing_stats = (tiled_kwargs or {}).get("timing_stats")
        if timing_stats is not None:
            timing_stats["pixel_conversion_s"] = (
                timing_stats.get("pixel_conversion_s", 0.0)
                + time.perf_counter() - conversion_started
            )
        return pred, img.shape[-2:]

    img = img.to(device)
    eval_size = model._cap_eval_size(img.shape[-2:])
    transformed = model.resize_and_pad_imgs_instance_panoptic([img])
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with autocast(dtype=torch.float16, device_type=device_type, enabled=device_type == "cuda"):
        outputs = model(transformed)
    masks = F.interpolate(outputs[0][-1].float(), model.img_size, mode="bilinear")
    masks = model.revert_resize_and_pad_logits_instance_panoptic(masks, [eval_size])
    pred = model.to_per_pixel_preds_panoptic(
        masks, outputs[1][-1], model.stuff_classes, model.mask_thresh, model.overlap_thresh
    )[0]
    return pred.cpu(), eval_size


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--infer_size", type=int, default=None)
    parser.add_argument(
        "--tile-config",
        "--tile_config",
        dest="tile_config",
        default=None,
        help="tile-expert config; defaults to --config when --tile-ckpt is set",
    )
    parser.add_argument(
        "--tile-ckpt",
        "--tile_ckpt",
        dest="tile_ckpt",
        default=None,
        help="expert checkpoint used only for tile raw-query forwards",
    )
    parser.add_argument(
        "--tile-infer-size",
        "--tile_infer_size",
        dest="tile_infer_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--tile-share-encoder",
        "--tile_share_encoder",
        dest="tile_share_encoder",
        action="store_true",
        help="share an exactly identical main/tile encoder to reduce memory; "
        "a last4/full expert is rejected",
    )
    parser.add_argument("--eval_max_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--index_start", type=int, default=0,
        help="inclusive dataset index for deterministic sharded evaluation",
    )
    parser.add_argument(
        "--index_end", type=int, default=None,
        help="exclusive dataset index for deterministic sharded evaluation",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="analyze this many indices spaced uniformly across the split",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--inference_mode",
        choices=["standard", "full-only", "tile-only", "full+tile"],
        default="standard",
        help="standard keeps the established capped eval; tiled modes use full-resolution GT",
    )
    parser.add_argument("--tile_size", type=int, nargs=2, default=[1120, 1120])
    parser.add_argument("--tile_overlap", type=float, default=0.2)
    parser.add_argument(
        "--tile_selector",
        choices=[
            "all",
            "oracle-small",
            "center",
            "horizon-row",
            "random",
            "query-thing",
            "query-small",
            "query-uncertainty",
        ],
        default="all",
        help="tile selection policy. 'oracle-small' is a GT upper bound; "
             "center/random are evidence-free equal-budget controls; the query "
             "policies reuse full-pass predictions and are deployable",
    )
    parser.add_argument(
        "--tile_budget", type=int, default=None,
        help="maximum tiles per image for any non-'all' selector",
    )
    parser.add_argument(
        "--tile_oracle_max_area", type=float, default=32**2,
        help="GT thing area cutoff used only by the oracle selector (default: COCO small)",
    )
    parser.add_argument(
        "--tile_random_seed", type=int, default=0,
        help="global seed used only by the deterministic random tile selector",
    )
    parser.add_argument(
        "--tile_router_top_fraction", type=float, default=0.01,
        help="fraction of strongest evidence pixels used to score each tile",
    )
    parser.add_argument(
        "--tile_router_min_query_score", type=float, default=0.05,
        help="minimum full-pass thing probability contributing routing evidence",
    )
    parser.add_argument(
        "--tile_router_resolution", choices=["fullres", "native"], default="fullres",
        help="score router evidence after legacy full-resolution projection or "
             "directly on the native query mask grid",
    )
    parser.add_argument(
        "--tile_router_min_tile_score", type=float, default=None,
        help="adaptive stop for query routing: do not select this or later tiles "
             "when the next tile evidence score is below this threshold. Unset "
             "preserves fixed-budget routing exactly",
    )
    parser.add_argument(
        "--tile_score_threshold", type=float, nargs="+", default=[0.1],
        help="tile-branch score threshold. One value applies to every thing "
             "class; pass one per thing class (ship obstacle) to threshold them "
             "asymmetrically -- at a shared 0.1 the tile branch pays +210 FP for "
             "+81 TP on obstacle but +252 FP for +132 TP plus +2.0 SQ on ship, "
             "so the two classes do not want the same operating point",
    )
    parser.add_argument("--tile_mask_threshold", type=float, default=0.5)
    parser.add_argument("--tile_local_nms_iou", type=float, default=0.7)
    parser.add_argument("--tile_merge_threshold", type=float, default=0.55)
    parser.add_argument("--tile_merge_mode", choices=["union", "best"], default="union")
    parser.add_argument("--tile_merge_metric", choices=["iou", "iomin"], default="iou")
    parser.add_argument("--tile_min_visible_fraction", type=float, default=0.25)
    parser.add_argument(
        "--tile_ownership_max_tile_area",
        type=float,
        default=None,
        help="enable conflict-aware ownership only for tile-only merged masks "
        "at or below this area (off by default)",
    )
    parser.add_argument(
        "--tile_ownership_min_tile_score",
        type=float,
        default=0.97,
        help="minimum tile score for conflict-aware ownership",
    )
    parser.add_argument(
        "--tile_ownership_min_full_tile_area_ratio",
        type=float,
        default=4.0,
        help="minimum conflicting full-mask area divided by tile-mask area",
    )
    parser.add_argument(
        "--tile_max_area", type=float, default=None,
        help="tile-branch FP gate: drop tile candidates above this pixel area "
             "(off by default; 9216 == 96^2 restricts tiles to small+medium)",
    )
    parser.add_argument(
        "--tile_sea_min_water", type=float, default=None,
        help="tile-branch FP gate: drop tile candidates whose surrounding band is "
             "less than this fraction seasurface (off by default)",
    )
    parser.add_argument("--tile_sea_class_id", type=int, default=2)
    parser.add_argument("--tile_sea_margin_frac", type=float, default=0.5)
    parser.add_argument(
        "--tile_gate_stage", choices=["before", "after"], default="before",
        help="gate raw per-tile candidates ('before') or the cross-tile merge "
             "result ('after'). 'after' judges whole instances, so a large object "
             "split across tiles fails the size gate as one object instead of "
             "passing it as fragments",
    )
    parser.add_argument(
        "--tile_protect_full_above_area", type=float, nargs="+", default=None,
        help="do not let tile fragments union into a full-pass mask already at "
             "least this pixel area (off by default). Targets the ship-large "
             "regression on val 622: recall 0.912 -> 0.889 where the extra "
             "misses are fn_poor_mask (39->60), not fn_no_overlap -- the object "
             "was found and the union stitch degraded its mask. One value "
             "applies to every thing class; pass one per thing class to set it "
             "asymmetrically (0 = off for that class) -- the union helps "
             "obstacle masks and hurts ship's",
    )
    parser.add_argument(
        "--no_pq", action="store_true",
        help="skip the official PanopticQuality channel (instance stats only)",
    )
    parser.add_argument(
        "--metrics_device", default=None,
        help="device for PQ and instance statistics (default: same as --device). "
             "Native 2800x2800 maps make CPU scoring ~60 s/image; 'cpu' is only "
             "useful when GPU memory is the binding constraint",
    )
    parser.add_argument(
        "--group_by_sequence", action="store_true",
        help="append per-sequence class/size rows to the JSON output",
    )
    parser.add_argument(
        "--csv_out", default=None,
        help="optionally write overall class/size rows as a stable CSV table",
    )
    parser.add_argument(
        "--compare", default=None, metavar="BASELINE_JSON",
        help="append candidate-minus-baseline deltas using a previous JSON output",
    )
    parser.add_argument(
        "--record_instance_matches",
        action="store_true",
        help="append one lightweight class/area/matched record per GT thing; "
        "used for reproducible arbitrary area-bin plots",
    )
    parser.add_argument(
        "--record_tile_details",
        action="store_true",
        help="append per-image candidate and selected tile windows; used for "
        "routing-position and union-coverage visualizations",
    )
    parser.add_argument("--supplement-spec", default=None, help="JSON opt-in audited experiment protocol")
    return parser


def main():
    args = build_parser().parse_args()
    supplement = json.loads(Path(args.supplement_spec).read_text()) if args.supplement_spec else {}
    sidecar = Path(args.out).with_suffix('')
    if supplement:
        from tools.reobs_core import json_dump
        sidecar.mkdir(parents=True,exist_ok=True)
        for stale in sidecar.glob('*.jsonl'):
            stale.unlink()
        json_dump(sidecar/'protocol.json',supplement)

    if args.tile_config is not None and args.tile_ckpt is None:
        raise SystemExit("--tile-config requires --tile-ckpt")
    if args.tile_share_encoder and args.tile_ckpt is None:
        raise SystemExit("--tile-share-encoder requires --tile-ckpt")
    if args.tile_ckpt is not None and args.inference_mode not in {
        "tile-only",
        "full+tile",
    }:
        raise SystemExit("--tile-ckpt requires a tiled inference mode")

    query_selectors = {"query-thing", "query-small", "query-uncertainty"}
    static_selectors = {"center", "random", "horizon-row"}
    if args.tile_selector != "all":
        if args.inference_mode not in {"tile-only", "full+tile"}:
            raise SystemExit("--tile_selector requires a tiled inference mode")
        if args.tile_budget is None or args.tile_budget <= 0:
            raise SystemExit("a non-'all' --tile_selector requires --tile_budget > 0")
        if args.tile_selector in query_selectors and args.inference_mode != "full+tile":
            raise SystemExit("query-based tile selectors require --inference_mode full+tile")
    elif args.tile_budget is not None:
        raise SystemExit("--tile_budget is only valid with a non-'all' --tile_selector")
    if args.tile_router_min_tile_score is not None:
        if args.tile_selector not in query_selectors:
            raise SystemExit(
                "--tile_router_min_tile_score requires a query-based --tile_selector"
            )
        if (
            not np.isfinite(args.tile_router_min_tile_score)
            or args.tile_router_min_tile_score < 0
        ):
            raise SystemExit(
                "--tile_router_min_tile_score must be finite and non-negative"
            )
    if args.tile_selector != "random" and args.tile_random_seed != 0:
        raise SystemExit("--tile_random_seed is only valid with --tile_selector random")

    from datasets.sea_panoptic import SeaPanopticDataset, parse_seq_frame
    from tools.eomt_infer import build_inference_model, share_encoder_if_identical

    device = args.device if torch.cuda.is_available() else "cpu"
    metrics_device = args.metrics_device or device
    if not torch.cuda.is_available():
        metrics_device = "cpu"
    model = build_inference_model(
        args.config, args.ckpt, device=device, infer_size=args.infer_size,
        return_query_embed=bool(supplement.get('tracking')),
        return_mask_pooled=bool(supplement.get('tracking')),
        reid_enabled=bool(supplement.get('tracking')),
        require_complete_network=bool(supplement),
        require_reid_head=bool(supplement.get('tracking')),
    )
    if supplement:
        model.supplement_options = supplement
        if supplement.get('backend')=='m2f':
            from tools.m2f_backend import install_backend
            install_backend(model,supplement,device)
        if 'full_threshold' in supplement:
            model.mask_thresh=float(supplement['full_threshold'])
        if supplement.get('trace',True):
            model.supplement_trace = {}
    tile_model = None
    if args.tile_ckpt is not None:
        tile_model = build_inference_model(
            args.tile_config or args.config,
            args.tile_ckpt,
            device="cpu" if args.tile_share_encoder else device,
            infer_size=args.tile_infer_size,
            require_complete_network=True,
        )
        if args.tile_share_encoder:
            tile_model = share_encoder_if_identical(model, tile_model).to(device).eval()
    if args.eval_max_size is not None:
        model.eval_max_size = args.eval_max_size
    dataset = SeaPanopticDataset(args.data, args.split, transforms=None, sequence_mode=bool(supplement), use_gt_track_ids=bool(supplement))
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    index_end = count if args.index_end is None else min(int(args.index_end), count)
    if args.index_start < 0 or args.index_start >= index_end:
        raise SystemExit(
            f"invalid index range [{args.index_start}, {index_end}) for {count} images"
        )
    available_indices = list(range(int(args.index_start), index_end))
    if args.sample and args.sample < len(available_indices):
        positions = np.linspace(
            0, len(available_indices) - 1, args.sample, dtype=int
        ).tolist()
        indices = [available_indices[position] for position in positions]
    else:
        indices = available_indices
    stats = defaultdict(lambda: defaultdict(float))
    sequence_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    sequence_images = defaultdict(int)
    gate_stats: dict = {}
    selection_stats: dict = {}
    timing_stats: dict = {}
    selected_indices_sink = []
    thresholds = args.tile_score_threshold
    if len(thresholds) == 1:
        thresholds = thresholds * len(THING_CLASSES)
    if len(thresholds) != len(THING_CLASSES):
        raise SystemExit(
            f"--tile_score_threshold takes 1 or {len(THING_CLASSES)} values, "
            f"got {len(thresholds)}"
        )
    protect = args.tile_protect_full_above_area
    if protect is not None:
        if len(protect) == 1:
            protect = protect * len(THING_CLASSES)
        if len(protect) != len(THING_CLASSES):
            raise SystemExit(
                f"--tile_protect_full_above_area takes 1 or {len(THING_CLASSES)} "
                f"values, got {len(protect)}"
            )
        protect = {c: v for c, v in zip(THING_CLASSES, protect) if v > 0} or None
    tiled_kwargs = {
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "score_thresholds": dict(zip(THING_CLASSES, thresholds)),
        "mask_threshold": args.tile_mask_threshold,
        "local_nms_iou": args.tile_local_nms_iou,
        "merge_iou": args.tile_merge_threshold,
        "merge_mode": args.tile_merge_mode,
        "merge_metric": args.tile_merge_metric,
        "min_visible_fraction": args.tile_min_visible_fraction,
        "ownership_max_tile_area": args.tile_ownership_max_tile_area,
        "ownership_min_tile_score": args.tile_ownership_min_tile_score,
        "ownership_min_full_tile_area_ratio": (
            args.tile_ownership_min_full_tile_area_ratio
        ),
        "tile_max_area": args.tile_max_area,
        "sea_min_water": args.tile_sea_min_water,
        "sea_class_id": args.tile_sea_class_id,
        "sea_margin_frac": args.tile_sea_margin_frac,
        "gate_stage": args.tile_gate_stage,
        "protect_full_above_area": protect,
        "stats": gate_stats,
        "timing_stats": timing_stats,
    }
    if args.tile_selector in query_selectors:
        tiled_kwargs.update({
            "tile_router": args.tile_selector,
            "tile_budget": args.tile_budget,
            "router_top_fraction": args.tile_router_top_fraction,
            "router_min_query_score": args.tile_router_min_query_score,
            "router_resolution": args.tile_router_resolution,
            "selected_indices_sink": selected_indices_sink,
        })
        if args.tile_router_min_tile_score is not None:
            tiled_kwargs["router_min_tile_score"] = (
                args.tile_router_min_tile_score
            )
    pq_metric = None if args.no_pq else init_pq_metric(model, metrics_device)
    latencies = []
    instance_records = []
    tile_records = []

    for position, index in enumerate(indices):
        img, target = dataset._build_sample(index) if supplement else dataset[index]
        if supplement.get('warmup',False) and position==0:
            from tools.eomt_infer import infer_panoptic_map
            infer_panoptic_map(model,img,device)
            if str(device).startswith('cuda'):torch.cuda.synchronize(device)

        image_tiled_kwargs = tiled_kwargs
        selection_diagnostics = None
        if args.tile_selector == "oracle-small":
            selected_tiles, selection_diagnostics = select_oracle_small_tiles(
                target,
                img.shape[-2],
                img.shape[-1],
                args.tile_size,
                args.tile_overlap,
                args.tile_budget,
                max_area=args.tile_oracle_max_area,
            )
            image_tiled_kwargs = {**tiled_kwargs, "tile_indices": selected_tiles}
        elif args.tile_selector in static_selectors:
            selected_tiles = select_static_tiles(
                img.shape[-2],
                img.shape[-1],
                args.tile_size,
                args.tile_overlap,
                args.tile_budget,
                args.tile_selector,
                seed=args.tile_random_seed,
                image_index=index,
            )
            image_tiled_kwargs = {**tiled_kwargs, "tile_indices": selected_tiles}
            selection_diagnostics = selected_small_tile_diagnostics(
                target,
                img.shape[-2],
                img.shape[-1],
                args.tile_size,
                args.tile_overlap,
                selected_tiles,
                max_area=args.tile_oracle_max_area,
            )
        elif args.tile_selector in query_selectors:
            selected_indices_sink.clear()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        pred, eval_size = predict(
            model,
            img,
            device,
            args.inference_mode,
            image_tiled_kwargs,
            tile_model=tile_model,
        )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - started)
        if args.tile_selector in query_selectors:
            selection_diagnostics = selected_small_tile_diagnostics(
                target,
                img.shape[-2],
                img.shape[-1],
                args.tile_size,
                args.tile_overlap,
                selected_indices_sink,
                max_area=args.tile_oracle_max_area,
            )
        if selection_diagnostics is not None:
            for key, value in selection_diagnostics.items():
                selection_stats[key] = selection_stats.get(key, 0) + value
        sample_path = dataset.samples[index][0]
        if supplement:
            from tools.reobs_core import json_dump
            json_dump(sidecar/'progress.json',{'completed':position+1,'total':len(indices),'last_index':index,'last_seconds':latencies[-1]})
        sequence, frame = parse_seq_frame(sample_path)
        if args.record_tile_details:
            from tile_geometry import sliding_tiles

            candidate_windows = sliding_tiles(
                img.shape[-2],
                img.shape[-1],
                args.tile_size,
                args.tile_overlap,
            )
            if args.tile_selector in query_selectors:
                recorded_indices = list(selected_indices_sink)
            elif args.tile_selector == "all":
                recorded_indices = list(range(len(candidate_windows))) if args.inference_mode != "full-only" else []
            else:
                recorded_indices = list(selected_tiles)
            tile_records.append(
                {
                    "image_index": int(index),
                    "file_name": str(sample_path),
                    "sequence": sequence,
                    "frame": int(frame),
                    "image_height": int(img.shape[-2]),
                    "image_width": int(img.shape[-1]),
                    "candidate_windows": [list(map(int, window)) for window in candidate_windows],
                    "selected_indices": [int(value) for value in recorded_indices],
                    "selected_windows": [
                        list(map(int, candidate_windows[value]))
                        for value in recorded_indices
                    ],
                }
            )
        gt = model.to_per_pixel_targets_panoptic([target])[0]
        gt = model._resize_per_pixel_target(gt, eval_size).cpu()
        pred, gt = pred.to(metrics_device), gt.to(metrics_device)
        if pq_metric is not None:
            model.update_metrics_panoptic(
                [pred], [gt], [target["is_crowd"].to(metrics_device)], -1
            )
        gt_instances = instances_from_map(gt, is_crowd=target["is_crowd"])
        ignored_regions = {cid: ~torch.isin(gt[..., 0], torch.tensor([0,1,2,3],device=gt.device)) for cid in THING_CLASSES}
        for instance_id in torch.nonzero(target["is_crowd"],as_tuple=False).flatten().tolist():
            cid = int(target["labels"][instance_id])
            if cid in ignored_regions:ignored_regions[cid] |= (gt[...,0]==cid) & (gt[...,1]==instance_id)
        pred_instances = instances_from_map(pred)
        update_stats(
            stats,
            gt_instances,
            pred_instances,
            instance_records if args.record_instance_matches else None,
            {
                "image_index": int(index),
                "file_name": str(sample_path),
                "sequence": sequence,
                "frame": int(frame),
                "image_height": int(img.shape[-2]),
                "image_width": int(img.shape[-1]),
            },
            ignored_regions=ignored_regions,
        )
        if supplement:
            from tools.reobs_core import record_frame
            from tile_geometry import sliding_tiles
            windows=sliding_tiles(img.shape[-2],img.shape[-1],args.tile_size,args.tile_overlap)
            chosen = list(selected_indices_sink) if args.tile_selector in query_selectors else list(selected_tiles) if args.tile_selector in static_selectors else list(range(len(windows))) if args.inference_mode!='full-only' else []
            if supplement.get('strict_budget') and args.tile_selector in query_selectors and len(chosen)!=min(args.tile_budget,len(windows)):
                raise RuntimeError('strict tile budget violated')
            trace=getattr(model,'supplement_trace',{})
            trace['save_candidates']=supplement.get('save_candidates',False)
            trace['tracking']=supplement.get('tracking',False)
            if supplement.get('trace',True):
                record_frame(sidecar,index,{'image_index':index,'file_name':str(sample_path),'sequence':sequence,'frame':int(frame),'image_height':int(img.shape[-2]),'image_width':int(img.shape[-1])},img,target,gt_instances,pred_instances,trace,chosen,windows,latencies[-1])
        if args.group_by_sequence:
            update_stats(sequence_stats[sequence], gt_instances, pred_instances, ignored_regions=ignored_regions)
            sequence_images[sequence] += 1
        if (position + 1) % 50 == 0 or position + 1 == len(indices):
            print(f"[error-analysis] {position + 1}/{len(indices)}", flush=True)

    rows = stats_rows(stats)
    pq = pq_summary(pq_metric) if pq_metric is not None else None

    output = {
        "supplement_protocol": supplement or None,
        "config": args.config,
        "checkpoint": args.ckpt,
        "tile_config": args.tile_config or (args.config if args.tile_ckpt else None),
        "tile_checkpoint": args.tile_ckpt,
        "tile_network_input": list(tile_model.img_size) if tile_model else None,
        "tile_shared_encoder": bool(args.tile_share_encoder),
        "split": args.split,
        "images": len(indices),
        "split_images": len(dataset),
        "index_range": [int(args.index_start), int(index_end)],
        "network_input": list(model.img_size),
        "eval_max_size": model.eval_max_size,
        "inference_mode": args.inference_mode,
        "metrics_device": str(metrics_device),
        "tile_params": (
            {
                **{
                    key: value
                    for key, value in tiled_kwargs.items()
                    if key not in {"stats", "timing_stats", "selected_indices_sink"}
                },
                "tile_selector": args.tile_selector,
                "tile_budget": args.tile_budget,
                "tile_oracle_max_area": args.tile_oracle_max_area,
                "tile_random_seed": (
                    args.tile_random_seed if args.tile_selector == "random" else None
                ),
                "tile_router_top_fraction": args.tile_router_top_fraction,
                "tile_router_min_query_score": args.tile_router_min_query_score,
                "tile_router_resolution": args.tile_router_resolution,
                **(
                    {
                        "tile_router_min_tile_score": (
                            args.tile_router_min_tile_score
                        )
                    }
                    if args.tile_router_min_tile_score is not None
                    else {}
                ),
            }
            if args.inference_mode in {"tile-only", "full+tile"}
            else None
        ),
        "tile_gate_stats": dict(sorted(gate_stats.items())) or None,
        "tile_selection_stats": (
            dict(sorted(selection_stats.items()))
            if args.tile_selector != "all"
            else None
        ),
        "latency_stages": (
            {
                key.replace("_s", "_mean_s"): value / timing_stats["images"]
                for key, value in sorted(timing_stats.items())
                if key.endswith("_s")
            }
            if timing_stats.get("images")
            else None
        ),
        "latency": (
            {
                "mean_s": sum(latencies) / len(latencies),
                "median_s": sorted(latencies)[len(latencies) // 2],
                "min_s": min(latencies),
                "max_s": max(latencies),
                "note": "per-image inference seconds, scoring excluded",
            }
            if latencies
            else None
        ),
        "pq": pq,
        "rows": rows,
    }
    if args.record_instance_matches:
        output["instance_records"] = instance_records
    if args.record_tile_details:
        output["tile_records"] = tile_records
    if args.group_by_sequence:
        output["sequence_images"] = dict(sorted(sequence_images.items()))
        output["sequence_rows"] = [
            row
            for sequence in sorted(sequence_stats)
            for row in stats_rows(sequence_stats[sequence], sequence=sequence)
        ]
    if args.compare:
        with Path(args.compare).open() as file:
            baseline_output = json.load(file)
        if not isinstance(baseline_output.get("rows"), list):
            raise ValueError(f"baseline JSON has no rows list: {args.compare}")
        output["comparison"] = {
            "baseline": str(args.compare),
            "rows": compare_rows(rows, baseline_output["rows"]),
            "pq": compare_pq(pq, baseline_output.get("pq")),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    if args.csv_out:
        write_csv(args.csv_out, rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
