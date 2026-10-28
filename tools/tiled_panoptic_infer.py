#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eomt_infer import (
    build_inference_model,
    infer_panoptic_map,
    infer_raw_query_detections,
    share_encoder_if_identical,
)
from tracking import maskutil
from tile_geometry import sliding_tiles


def select_tile_windows(
    windows: Sequence[Tuple[int, int, int, int]],
    tile_indices: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int, int, int]]:
    if tile_indices is None:
        return list(windows)
    indices = [int(index) for index in tile_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("tile_indices must not contain duplicates")
    invalid = [index for index in indices if index < 0 or index >= len(windows)]
    if invalid:
        raise ValueError(
            f"tile index out of range for {len(windows)} windows: {invalid}"
        )
    return [windows[index] for index in indices]


def select_density(centers, windows, k):
    remaining = set(range(len(centers)))
    selected = []
    for _ in range(min(k, len(windows))):
        hits = {i: {j for j in remaining if w[1] <= centers[j][0] < w[3] and w[0] <= centers[j][1] < w[2]} for i, w in enumerate(windows) if i not in selected}
        chosen = max(hits, key=lambda i: (len(hits[i]), -i))
        selected.append(chosen)
        remaining -= hits[chosen]
    return selected


def select_tiles_by_evidence(
    evidence: np.ndarray,
    windows: Sequence[Tuple[int, int, int, int]],
    budget: int,
    top_fraction: float = 0.01,
    min_score: Optional[float] = None,
    diagnostics: Optional[Dict] = None,
) -> Tuple[List[int], List[float]]:
    if evidence.ndim != 2:
        raise ValueError("routing evidence must be a 2-D array")
    if budget <= 0:
        raise ValueError("tile routing budget must be positive")
    if not 0 < top_fraction <= 1:
        raise ValueError("router top_fraction must be in (0, 1]")
    if min_score is not None and (
        not math.isfinite(float(min_score)) or float(min_score) < 0
    ):
        raise ValueError("router min_score must be finite and non-negative")

    residual = np.asarray(evidence, dtype=np.float32).copy()
    selected: List[int] = []
    selected_scores: List[float] = []
    candidate_slots = min(int(budget), len(windows))
    stop_reason = "budget"
    stop_score: Optional[float] = None
    for _ in range(candidate_slots):
        scores = np.full(len(windows), -np.inf, dtype=np.float64)
        for index, (y0, x0, y1, x1) in enumerate(windows):
            if index in selected:
                continue
            values = residual[y0:y1, x0:x1].reshape(-1)
            top_n = max(1, int(round(values.size * top_fraction)))
            if top_n < values.size:
                score = float(np.partition(values, values.size - top_n)[-top_n:].mean())
            else:
                score = float(values.mean())
            scores[index] = score
        best_index = int(np.argmax(scores)) if len(scores) else -1
        if (
            best_index < 0
            or not np.isfinite(scores[best_index])
            or scores[best_index] <= 0
        ):
            stop_reason = "no-positive-evidence"
            break
        if min_score is not None and float(scores[best_index]) < float(min_score):
            stop_reason = "adaptive-threshold"
            stop_score = float(scores[best_index])
            break
        selected.append(best_index)
        selected_scores.append(float(scores[best_index]))
        y0, x0, y1, x1 = windows[best_index]
        residual[y0:y1, x0:x1] = 0
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "stop_reason": stop_reason,
                "stop_score": stop_score,
                "candidate_slots": candidate_slots,
                "selected_tiles": len(selected),
                "adaptive_tiles_saved": (
                    candidate_slots - len(selected)
                    if stop_reason == "adaptive-threshold"
                    else 0
                ),
            }
        )
    return selected, selected_scores


def project_windows_to_evidence_grid(
    windows: Sequence[Tuple[int, int, int, int]],
    image_size: Sequence[int],
    scaled_size: Sequence[int],
    padded_size: Sequence[int],
    evidence_size: Sequence[int],
) -> List[Tuple[int, int, int, int]]:
    image_h, image_w = (int(v) for v in image_size)
    scaled_h, scaled_w = (int(v) for v in scaled_size)
    padded_h, padded_w = (int(v) for v in padded_size)
    evidence_h, evidence_w = (int(v) for v in evidence_size)
    if min(image_h, image_w, scaled_h, scaled_w, padded_h, padded_w) <= 0:
        raise ValueError("image/scaled/padded sizes must be positive")

    scale_y = scaled_h * evidence_h / (image_h * padded_h)
    scale_x = scaled_w * evidence_w / (image_w * padded_w)
    projected = []
    for y0, x0, y1, x1 in windows:
        ey0 = max(0, min(evidence_h - 1, math.floor(y0 * scale_y)))
        ex0 = max(0, min(evidence_w - 1, math.floor(x0 * scale_x)))
        ey1 = max(ey0 + 1, min(evidence_h, math.ceil(y1 * scale_y)))
        ex1 = max(ex0 + 1, min(evidence_w, math.ceil(x1 * scale_x)))
        projected.append((ey0, ex0, ey1, ex1))
    return projected


def select_tiles_by_native_evidence(
    routing_payload: Dict,
    windows: Sequence[Tuple[int, int, int, int]],
    budget: int,
    top_fraction: float = 0.01,
    min_score: Optional[float] = None,
    diagnostics: Optional[Dict] = None,
) -> Tuple[List[int], List[float]]:
    if budget <= 0:
        raise ValueError("tile routing budget must be positive")
    if not 0 < top_fraction <= 1:
        raise ValueError("router top_fraction must be in (0, 1]")
    if min_score is not None and (
        not math.isfinite(float(min_score)) or float(min_score) < 0
    ):
        raise ValueError("router min_score must be finite and non-negative")
    evidence = routing_payload.get("evidence")
    if not isinstance(evidence, torch.Tensor) or evidence.ndim != 2:
        raise ValueError("native routing payload needs a 2-D evidence tensor")
    projected = project_windows_to_evidence_grid(
        windows,
        routing_payload["image_size"],
        routing_payload["scaled_size"],
        routing_payload["padded_size"],
        evidence.shape,
    )

    residual = evidence.float().clone()
    selected: List[int] = []
    selected_scores: List[float] = []
    candidate_slots = min(int(budget), len(windows))
    stop_reason = "budget"
    stop_score: Optional[float] = None
    for _ in range(candidate_slots):
        scores = torch.full(
            (len(windows),), -torch.inf, device=residual.device, dtype=residual.dtype
        )
        for index, (y0, x0, y1, x1) in enumerate(projected):
            if index in selected:
                continue
            values = residual[y0:y1, x0:x1].reshape(-1)
            top_n = max(1, int(round(values.numel() * top_fraction)))
            scores[index] = torch.topk(values, min(top_n, values.numel())).values.mean()
        best_index = int(torch.argmax(scores).item()) if len(scores) else -1
        best_score = float(scores[best_index].item()) if best_index >= 0 else 0.0
        if best_index < 0 or best_score <= 0:
            stop_reason = "no-positive-evidence"
            break
        if min_score is not None and best_score < float(min_score):
            stop_reason = "adaptive-threshold"
            stop_score = best_score
            break
        selected.append(best_index)
        selected_scores.append(best_score)
        y0, x0, y1, x1 = projected[best_index]
        residual[y0:y1, x0:x1] = 0
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "stop_reason": stop_reason,
                "stop_score": stop_score,
                "candidate_slots": candidate_slots,
                "selected_tiles": len(selected),
                "adaptive_tiles_saved": (
                    candidate_slots - len(selected)
                    if stop_reason == "adaptive-threshold"
                    else 0
                ),
            }
        )
    return selected, selected_scores


def update_router_diagnostic_stats(
    stats: Dict,
    selected_scores: Sequence[float],
    diagnostics: Dict,
    adaptive_stop_enabled: bool,
) -> None:
    stats["router_score_sum"] = stats.get("router_score_sum", 0.0) + sum(
        selected_scores
    )
    if not selected_scores and diagnostics.get("stop_reason") == "no-positive-evidence":
        stats["router_zero_evidence_images"] = (
            stats.get("router_zero_evidence_images", 0) + 1
        )
    if not adaptive_stop_enabled:
        return

    selected_count = len(selected_scores)
    histogram_key = f"router_images_selecting_{selected_count}_tiles"
    stats[histogram_key] = stats.get(histogram_key, 0) + 1
    stats.setdefault("router_adaptive_stop_images", 0)
    stats.setdefault("router_adaptive_tiles_saved", 0)
    stats.setdefault("router_adaptive_stop_score_sum", 0.0)
    if diagnostics.get("stop_reason") != "adaptive-threshold":
        return

    stop_score = float(diagnostics["stop_score"])
    stats["router_adaptive_stop_images"] += 1
    stats["router_adaptive_tiles_saved"] += int(diagnostics["adaptive_tiles_saved"])
    stats["router_adaptive_stop_score_sum"] += stop_score
    stats["router_adaptive_stop_score_min"] = min(
        stats.get("router_adaptive_stop_score_min", stop_score), stop_score
    )
    stats["router_adaptive_stop_score_max"] = max(
        stats.get("router_adaptive_stop_score_max", stop_score), stop_score
    )


def _decode(det: Dict) -> np.ndarray:
    return np.asarray(maskutil.decode(det["mask_rle"]), dtype=bool)


def _per_class_arg(values, thing_classes, flag):
    if values is None:
        return None
    if len(values) == 1:
        values = list(values) * len(thing_classes)
    if len(values) != len(thing_classes):
        raise SystemExit(
            f"{flag} takes 1 or {len(thing_classes)} values "
            f"(one per --thing-classes entry), got {len(values)}"
        )
    return {c: v for c, v in zip(thing_classes, values) if v > 0}


def _class_value(spec, class_id):
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec.get(int(class_id))
    return spec


def merge_same_class_instances(
    detections: Sequence[Dict],
    iou_threshold: float = 0.55,
    fuse: str = "union",
    overlap_metric: str = "iou",
    protect_full_above_area: Optional[float] = None,
) -> List[Dict]:
    if fuse not in {"union", "best"}:
        raise ValueError("fuse must be 'union' or 'best'")
    if overlap_metric not in {"iou", "iomin"}:
        raise ValueError("overlap_metric must be 'iou' or 'iomin'")
    pending = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
    merged: List[Dict] = []
    for det in pending:
        mask = _decode(det)
        mask_area = int(mask.sum())
        is_tile = det.get("source") == "tile"
        match = None
        for i, old in enumerate(merged):
            if int(old["class_id"]) != int(det["class_id"]):
                continue
            old_mask = old["_mask"]
            inter = int(np.logical_and(mask, old_mask).sum())
            denominator = (
                mask_area + old["_area"] - inter
                if overlap_metric == "iou"
                else min(mask_area, old["_area"])
            )
            if denominator and inter / denominator >= iou_threshold:
                match = i
                break
        if match is None:
            item = dict(det)
            item["_mask"] = mask
            item["_area"] = mask_area
            item["_full"] = None if is_tile else mask.copy()
            item["_full_area"] = None if is_tile else mask_area
            merged.append(item)
        elif fuse == "union":
            target = merged[match]
            if "lineage" in target or "lineage" in det:
                target["lineage"] = target.get("lineage", []) + det.get("lineage", [])
            np.logical_or(target["_mask"], mask, out=target["_mask"])
            target["_area"] += mask_area - inter
            if not is_tile:
                if target["_full"] is None:
                    target["_full"] = mask.copy()
                    target["_full_area"] = mask_area
                else:
                    full_inter = int(np.logical_and(target["_full"], mask).sum())
                    np.logical_or(target["_full"], mask, out=target["_full"])
                    target["_full_area"] += mask_area - full_inter
                target["source"] = det.get("source", "full")

    result = []
    for item in merged:
        mask = item.pop("_mask")
        mask_area = item.pop("_area")
        full = item.pop("_full")
        full_area = item.pop("_full_area")
        limit = _class_value(protect_full_above_area, item["class_id"])
        if limit is not None and full is not None and float(full_area) >= float(limit):
            mask = full
            mask_area = full_area
        item["mask_rle"] = maskutil.encode(mask)
        item["area"] = float(mask_area)
        x, y, bw, bh = maskutil.to_bbox(item["mask_rle"])
        item["bbox"] = [x, y, x + bw, y + bh]
        item["centroid"] = [x + bw / 2, y + bh / 2]
        item["aspect_ratio"] = float(bw / max(bh, 1e-3))
        result.append(item)
    return result


def sea_mask_from_panoptic(
    full_maps: Optional[Tuple[np.ndarray, np.ndarray]], sea_class_id: int
) -> Optional[np.ndarray]:
    if full_maps is None:
        return None
    cls_map, _ = full_maps
    sea = cls_map == int(sea_class_id)
    return sea if sea.any() else None


def gate_tile_detections(
    detections: Sequence[Dict],
    max_area: Optional[float] = None,
    sea: Optional[np.ndarray] = None,
    min_water: Optional[float] = None,
    sea_margin_frac: float = 0.5,
) -> Tuple[List[Dict], Dict[str, int]]:
    if max_area is None and min_water is None:
        return list(detections), {"gated_area": 0, "gated_water": 0}
    if min_water is not None and sea is not None:
        from tracking.sea_prior import water_proximity

    kept: List[Dict] = []
    counts = {"gated_area": 0, "gated_water": 0}
    for det in detections:
        if det.get("source") != "tile":
            kept.append(det)
            continue
        if max_area is not None and float(det.get("area", 0.0)) > float(max_area):
            counts["gated_area"] += 1
            continue
        if min_water is not None and sea is not None:
            proximity = water_proximity(det["bbox"], _decode(det), sea, sea_margin_frac)
            if proximity < float(min_water):
                counts["gated_water"] += 1
                continue
        kept.append(det)
    return kept, counts


def things_from_panoptic(
    full_maps: Tuple[np.ndarray, np.ndarray],
    thing_classes: Sequence[int],
    segment_metadata: Optional[Dict[int, Dict]] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    cls_map, seg_map = full_maps
    segment_metadata = segment_metadata or {}
    class_names = class_names or {}
    detections = []
    for sid in np.unique(seg_map):
        if sid < 0:
            continue
        mask = seg_map == sid
        class_id = int(cls_map[mask][0])
        if class_id not in thing_classes:
            continue
        detection = {
            "class_id": class_id,
            "class_name": class_names.get(class_id, str(class_id)),
            "score": 1.0,
            "area": float(mask.sum()),
            "mask_rle": maskutil.encode(mask),
            "source": "full_panoptic",
        }
        detection.update(segment_metadata.get(int(sid), {}))
        detections.append(detection)
    return detections


def _validate_tile_ownership(
    max_tile_area: Optional[float],
    min_tile_score: float,
    min_full_tile_area_ratio: float,
) -> None:
    if max_tile_area is None:
        return
    if not math.isfinite(float(max_tile_area)) or float(max_tile_area) <= 0:
        raise ValueError("ownership_max_tile_area must be finite and positive")
    if not math.isfinite(float(min_tile_score)) or not 0 <= float(min_tile_score) <= 1:
        raise ValueError("ownership_min_tile_score must be finite and in [0, 1]")
    if (
        not math.isfinite(float(min_full_tile_area_ratio))
        or float(min_full_tile_area_ratio) <= 1
    ):
        raise ValueError(
            "ownership_min_full_tile_area_ratio must be finite and greater than 1"
        )


def _conflict_aware_compose_entries(
    things: Sequence[Dict],
    shape: Sequence[int],
    min_visible_fraction: float,
    max_tile_area: float,
    min_tile_score: float,
    min_full_tile_area_ratio: float,
) -> List[Tuple[Dict, np.ndarray, int]]:
    h, w = int(shape[0]), int(shape[1])
    ordered = sorted(things, key=lambda d: float(d["score"]), reverse=True)
    entries = [
        {"det": det, "mask": _decode(det), "position": position}
        for position, det in enumerate(ordered)
    ]
    claimed = np.zeros((h, w), dtype=bool)
    accepted: List[Dict] = []
    promotions: Dict[int, Tuple[int, np.ndarray, int]] = {}

    for entry in entries:
        det = entry["det"]
        mask = entry["mask"]
        area = int(mask.sum())
        visible_area = int(np.logical_and(mask, ~claimed).sum())
        passes = area > 0 and visible_area / area >= min_visible_fraction

        if (
            not passes
            and det.get("source") == "tile"
            and 0 < area <= float(max_tile_area)
            and float(det["score"]) >= float(min_tile_score)
        ):
            blockers = []
            nonblocking_claimed = np.zeros((h, w), dtype=bool)
            for previous in accepted:
                previous_mask = previous["mask"]
                previous_det = previous["det"]
                previous_area = int(previous["area"])
                is_large_full = (
                    previous_det.get("source") in {"full", "full_panoptic"}
                    and previous_area / area >= float(min_full_tile_area_ratio)
                    and bool(np.logical_and(mask, previous_mask).any())
                )
                if is_large_full:
                    blockers.append(previous)
                else:
                    np.logical_or(
                        nonblocking_claimed,
                        previous_mask,
                        out=nonblocking_claimed,
                    )

            restored_mask = np.logical_and(mask, ~nonblocking_claimed)
            restored_area = int(restored_mask.sum())
            if blockers and restored_area / area >= min_visible_fraction:
                insert_at = min(int(blocker["position"]) for blocker in blockers)
                promotions[int(entry["position"])] = (
                    insert_at,
                    restored_mask,
                    area,
                )

        if passes:
            accepted_entry = {**entry, "area": area}
            accepted.append(accepted_entry)
            np.logical_or(claimed, mask, out=claimed)

    promoted_at: Dict[int, List[Tuple[Dict, np.ndarray, int]]] = {}
    for position, (insert_at, mask, original_area) in promotions.items():
        promoted_at.setdefault(insert_at, []).append(
            (entries[position]["det"], mask, original_area)
        )

    result: List[Tuple[Dict, np.ndarray, int]] = []
    for position, entry in enumerate(entries):
        result.extend(promoted_at.get(position, []))
        if position not in promotions:
            mask = entry["mask"]
            result.append((entry["det"], mask, int(mask.sum())))
    return result


def compose_panoptic(
    full_maps: Optional[Tuple[np.ndarray, np.ndarray]],
    things: Sequence[Dict],
    thing_classes: Sequence[int],
    shape: Sequence[int],
    min_visible_fraction: float = 0.25,
    ownership_max_tile_area: Optional[float] = None,
    ownership_min_tile_score: float = 0.97,
    ownership_min_full_tile_area_ratio: float = 4.0,
    detections_sink: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    if detections_sink is not None:
        detections_sink.clear()
    _validate_tile_ownership(
        ownership_max_tile_area,
        ownership_min_tile_score,
        ownership_min_full_tile_area_ratio,
    )
    h, w = int(shape[0]), int(shape[1])
    pan = np.zeros((h, w), dtype=np.int32)
    segments: List[Dict] = []
    next_id = 1
    if full_maps is not None:
        cls_map, seg_map = full_maps
        for sid in np.unique(seg_map):
            if sid < 0:
                continue
            region = seg_map == sid
            cid = int(cls_map[region][0])
            if cid in thing_classes:
                continue
            pan[region] = next_id
            segments.append(
                {
                    "id": next_id,
                    "category_id": cid,
                    "area": int(region.sum()),
                    "iscrowd": 0,
                }
            )
            next_id += 1

    if ownership_max_tile_area is None:
        compose_entries = []
        for det in sorted(things, key=lambda d: float(d["score"]), reverse=True):
            mask = _decode(det)
            compose_entries.append((det, mask, int(mask.sum())))
    else:
        compose_entries = _conflict_aware_compose_entries(
            things,
            (h, w),
            min_visible_fraction,
            ownership_max_tile_area,
            ownership_min_tile_score,
            ownership_min_full_tile_area_ratio,
        )

    thing_claimed = np.zeros((h, w), dtype=bool)
    for det, mask, original_area in compose_entries:
        visible = mask & ~thing_claimed
        visible_area = int(visible.sum())
        if original_area == 0 or visible_area / original_area < min_visible_fraction:
            continue
        pan[visible] = next_id
        thing_claimed[visible] = True
        segments.append(
            {
                "id": next_id,
                "category_id": int(det["class_id"]),
                "area": visible_area,
                "iscrowd": 0,
                "score": float(det["score"]),
            }
        )
        if detections_sink is not None:
            rle = maskutil.encode(visible)
            x, y, width, height = maskutil.to_bbox(rle)
            tracking_score = float(det.get("tracking_score", det["score"]))
            exported = dict(det)
            exported.update(
                {
                    "score": tracking_score,
                    "composition_score": float(det["score"]),
                    "mask_rle": rle,
                    "bbox": [x, y, x + width, y + height],
                    "centroid": [x + width / 2, y + height / 2],
                    "area": float(visible_area),
                    "aspect_ratio": float(width / max(height, 1e-3)),
                    "panoptic_segment_id": next_id,
                }
            )
            exported.pop("tracking_score", None)
            detections_sink.append(exported)
        next_id += 1
    final_areas = np.bincount(pan.ravel(), minlength=next_id)
    corrected = []
    for segment in segments:
        final_area = int(final_areas[segment["id"]])
        if final_area:
            segment["area"] = final_area
            corrected.append(segment)
    return pan, corrected


@torch.no_grad()
def infer_tiled_panoptic(
    model,
    image: torch.Tensor,
    device: str,
    mode: str = "full-only",
    thing_classes: Sequence[int] = (0, 1),
    class_names: Optional[Dict[int, str]] = None,
    score_thresholds: Optional[Dict[int, float]] = None,
    tile_size: Sequence[int] = (1120, 1120),
    tile_overlap: float = 0.2,
    mask_threshold: float = 0.5,
    local_nms_iou: float = 0.7,
    merge_iou: float = 0.55,
    merge_mode: str = "union",
    merge_metric: str = "iou",
    min_visible_fraction: float = 0.25,
    ownership_max_tile_area: Optional[float] = None,
    ownership_min_tile_score: float = 0.97,
    ownership_min_full_tile_area_ratio: float = 4.0,
    tile_max_area: Optional[float] = None,
    sea_min_water: Optional[float] = None,
    sea_class_id: int = 2,
    sea_margin_frac: float = 0.5,
    gate_stage: str = "before",
    protect_full_above_area: Optional[float] = None,
    tile_indices: Optional[Sequence[int]] = None,
    tile_router: Optional[str] = None,
    tile_budget: Optional[int] = None,
    router_top_fraction: float = 0.01,
    router_min_query_score: float = 0.05,
    router_resolution: str = "fullres",
    router_min_tile_score: Optional[float] = None,
    selected_indices_sink: Optional[List[int]] = None,
    routing_evidence_sink: Optional[Dict] = None,
    stats: Optional[Dict[str, float]] = None,
    timing_stats: Optional[Dict[str, float]] = None,
    tile_model=None,
    want_reid: bool = False,
    detections_sink: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    if mode not in {"full-only", "tile-only", "full+tile"}:
        raise ValueError(f"unsupported mode: {mode}")
    if detections_sink is not None and mode == "full-only":
        raise ValueError(
            "detections_sink currently requires tile-only or full+tile mode"
        )
    if want_reid and detections_sink is None:
        raise ValueError("want_reid requires detections_sink")
    detection_model = model if tile_model is None else tile_model
    if mode != "full-only" and detection_model is not model:
        main_classes = getattr(model, "num_classes", None)
        tile_classes = getattr(detection_model, "num_classes", None)
        main_size = tuple(getattr(model, "img_size", ()))
        tile_size_model = tuple(getattr(detection_model, "img_size", ()))
        main_stuff = tuple(getattr(model, "stuff_classes", ()))
        tile_stuff = tuple(getattr(detection_model, "stuff_classes", ()))
        if (
            main_classes != tile_classes
            or main_size != tile_size_model
            or main_stuff != tile_stuff
        ):
            raise ValueError(
                "tile model must share num_classes, img_size, and stuff_classes "
                f"with the main model; got ({tile_classes}, {tile_size_model}, "
                f"{tile_stuff}) versus ({main_classes}, {main_size}, {main_stuff})"
            )
    if gate_stage not in {"before", "after"}:
        raise ValueError("gate_stage must be 'before' or 'after'")
    if tile_router not in {
        None,
        "query-thing",
        "query-small",
        "query-uncertainty",
    }:
        raise ValueError(f"unsupported tile router: {tile_router}")
    if router_resolution not in {"fullres", "native"}:
        raise ValueError("router_resolution must be 'fullres' or 'native'")
    _validate_tile_ownership(
        ownership_max_tile_area,
        ownership_min_tile_score,
        ownership_min_full_tile_area_ratio,
    )
    if tile_router is not None:
        if mode != "full+tile":
            raise ValueError("query tile routing requires full+tile mode")
        if tile_indices is not None:
            raise ValueError("tile_indices and tile_router are mutually exclusive")
        if tile_budget is None or tile_budget <= 0:
            raise ValueError("query tile routing requires a positive tile_budget")
    elif router_min_tile_score is not None:
        raise ValueError("router_min_tile_score requires query tile routing")
    if routing_evidence_sink is not None and tile_router is None:
        raise ValueError("routing_evidence_sink requires query tile routing")
    if router_min_tile_score is not None and (
        not math.isfinite(float(router_min_tile_score))
        or float(router_min_tile_score) < 0
    ):
        raise ValueError(
            "router_min_tile_score must be finite and non-negative when set"
        )
    h, w = image.shape[-2:]
    class_names = class_names or {int(c): str(c) for c in thing_classes}
    score_thresholds = score_thresholds or {int(c): 0.1 for c in thing_classes}
    options = getattr(model, "supplement_options", {})
    trace = getattr(model, "supplement_trace", None)
    if trace is not None:
        trace.clear()
        trace.update({"windows": [], "raw_windows": [], "merged": [], "final": []})
    if options.get("tracking") and detections_sink is None and mode != "full-only":
        detections_sink = trace["final"]
    stage_started = time.perf_counter()
    full_segment_metadata: Dict[int, Dict] = {}
    full_kwargs = {
        "routing_mode": tile_router,
        "routing_thing_classes": thing_classes,
        "routing_min_query_score": router_min_query_score,
        "routing_resolution": router_resolution,
    }
    if detections_sink is not None or trace is not None:
        full_kwargs["thing_query_metadata_sink"] = full_segment_metadata
    full_output = (
        infer_panoptic_map(model, image, device, **full_kwargs)
        if mode != "tile-only"
        else None
    )
    routing_evidence = full_output[2] if tile_router is not None else None
    if routing_evidence_sink is not None:
        routing_evidence_sink.clear()
        if isinstance(routing_evidence, dict):
            routing_evidence_sink.update(
                {
                    key: (
                        value.detach().float().cpu()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in routing_evidence.items()
                }
            )
        else:
            routing_evidence_sink["evidence"] = np.asarray(routing_evidence).copy()
            routing_evidence_sink["image_size"] = (int(h), int(w))
    full_maps = full_output[:2] if full_output is not None else None
    if trace is not None and full_maps is not None:
        trace["full_network_input"] = list(model.img_size)
        trace["full"] = things_from_panoptic(full_maps, thing_classes, segment_metadata=full_segment_metadata, class_names=class_names)
        if options.get("tracking"):
            trace["seasurface_rle"] = maskutil.encode(full_maps[0] == 2)
    full_pass_s = time.perf_counter() - stage_started

    if mode == "full-only":
        cls_map, seg_map = full_maps
        pan = np.zeros((h, w), dtype=np.int32)
        info, next_id = [], 1
        for sid in np.unique(seg_map):
            if sid < 0:
                continue
            region = seg_map == sid
            cid = int(cls_map[region][0])
            pan[region] = next_id
            info.append(
                {
                    "id": next_id,
                    "category_id": cid,
                    "area": int(region.sum()),
                    "iscrowd": 0,
                }
            )
            next_id += 1
        if timing_stats is not None:
            timing_stats["images"] = timing_stats.get("images", 0) + 1
            timing_stats["full_pass_s"] = (
                timing_stats.get("full_pass_s", 0.0) + full_pass_s
            )
            timing_stats["postprocess_s"] = timing_stats.get("postprocess_s", 0.0) + (
                time.perf_counter() - stage_started - full_pass_s
            )
        return pan, info

    full_thing_started = time.perf_counter()
    detections: List[Dict] = (
        things_from_panoptic(
            full_maps,
            thing_classes,
            segment_metadata=full_segment_metadata,
            class_names=class_names,
        )
        if mode == "full+tile"
        else []
    )
    all_windows = sliding_tiles(h, w, tile_size, tile_overlap)
    if options.get("density", False):
        centers = []
        for det in detections:
            x, y, bw, bh = maskutil.to_bbox(det["mask_rle"])
            centers.append((x + bw / 2, y + bh / 2))
        tile_indices = select_density(centers, all_windows, 3)
    full_thing_extract_s = time.perf_counter() - full_thing_started
    router_scores: List[float] = []
    router_diagnostics: Dict = {}
    router_started = time.perf_counter()
    if tile_router is not None and not options.get("density", False):
        if options:
            if router_resolution != "native":
                raise ValueError("supplement protocol requires native routing")
            from tools.reobs_core import select_supplement_tiles
            tile_indices, router_scores = select_supplement_tiles(routing_evidence, all_windows, tile_budget, router_top_fraction, router_min_tile_score, router_diagnostics, options)
        elif router_resolution == "native":
            tile_indices, router_scores = select_tiles_by_native_evidence(
                routing_evidence,
                all_windows,
                tile_budget,
                router_top_fraction,
                router_min_tile_score,
                router_diagnostics,
            )
        else:
            tile_indices, router_scores = select_tiles_by_evidence(
                routing_evidence,
                all_windows,
                tile_budget,
                router_top_fraction,
                router_min_tile_score,
                router_diagnostics,
            )
    router_s = time.perf_counter() - router_started
    if selected_indices_sink is not None:
        selected_indices_sink.clear()
        selected_indices_sink.extend(
            range(len(all_windows)) if tile_indices is None else tile_indices
        )
    selected_windows = select_tile_windows(all_windows, tile_indices)
    if trace is not None:
        trace["candidate_windows"] = [list(w) for w in all_windows]
        trace["selected_indices"] = list(range(len(all_windows))) if tile_indices is None else list(tile_indices)
        trace["router"] = router_diagnostics
        for d in detections:
            d["lineage"] = [{"source": "full", "query_index": d.get("query_index")}]
    tile_started = time.perf_counter()
    for y0, x0, y1, x1 in selected_windows:
        window_id = all_windows.index((y0, x0, y1, x1))
        raw_trace = [] if trace is not None and options.get("raw_trace") else None
        for det in infer_raw_query_detections(
            detection_model,
            image[:, y0:y1, x0:x1],
            device,
            thing_classes,
            class_names,
            score_thresholds,
            mask_threshold,
            local_nms_iou,
            want_reid=want_reid or options.get("tracking", False),
            want_query_embed=options.get("tracking", False),
            want_mask_pooled=options.get("tracking", False),
            raw_trace_sink=raw_trace,
            want_tracking_score=detections_sink is not None,
            xy_offset=(x0, y0),
            canvas_size=(h, w),
        ):
            det["source"] = "tile"
            if trace is not None:
                det["lineage"] = [{"source": "tile", "window_id": window_id, "query_index": det.get("query_index")}]
            detections.append(det)
        if trace is not None:
            trace["windows"].append({"window_id": window_id, "bounds": [y0, x0, y1, x1], "candidates": [dict(d) for d in detections if d.get("lineage", [{}])[0].get("window_id") == window_id]})
            if raw_trace is not None:
                trace["raw_windows"].append({"window_id": window_id, "bounds": [y0, x0, y1, x1], "queries": raw_trace})
    tile_forward_s = time.perf_counter() - tile_started
    postprocess_started = time.perf_counter()
    n_tile = sum(1 for d in detections if d.get("source") == "tile")
    sea = sea_mask_from_panoptic(full_maps, sea_class_id)
    gate = lambda dets: gate_tile_detections(
        dets,
        max_area=tile_max_area,
        sea=sea,
        min_water=sea_min_water,
        sea_margin_frac=sea_margin_frac,
    )
    if gate_stage == "before":
        detections, gate_counts = gate(detections)
        merge_started = time.perf_counter()
        detections = merge_same_class_instances(
            detections, merge_iou, merge_mode, merge_metric, protect_full_above_area
        )
        merge_s = time.perf_counter() - merge_started
    else:
        merge_started = time.perf_counter()
        detections = merge_same_class_instances(
            detections, merge_iou, merge_mode, merge_metric, protect_full_above_area
        )
        merge_s = time.perf_counter() - merge_started
        detections, gate_counts = gate(detections)
    if stats is not None:
        stats["available_tiles"] = stats.get("available_tiles", 0) + len(all_windows)
        stats["selected_tiles"] = stats.get("selected_tiles", 0) + len(selected_windows)
        if not selected_windows:
            stats["images_without_selected_tiles"] = (
                stats.get("images_without_selected_tiles", 0) + 1
            )
        if tile_router is not None:
            update_router_diagnostic_stats(
                stats,
                router_scores,
                router_diagnostics,
                adaptive_stop_enabled=router_min_tile_score is not None,
            )
        stats["tile_candidates"] = stats.get("tile_candidates", 0) + n_tile
        for key, value in gate_counts.items():
            stats[key] = stats.get(key, 0) + value
        stats["merged_things"] = stats.get("merged_things", 0) + len(detections)
    if trace is not None:
        trace["merged"] = [dict(d) for d in detections]
        if detections_sink is None:
            detections_sink = trace["final"]
    compose_started = time.perf_counter()
    result = compose_panoptic(
        full_maps,
        detections,
        thing_classes,
        (h, w),
        min_visible_fraction,
        ownership_max_tile_area,
        ownership_min_tile_score,
        ownership_min_full_tile_area_ratio,
        detections_sink=detections_sink,
    )
    compose_s = time.perf_counter() - compose_started
    if timing_stats is not None:
        timing_stats["images"] = timing_stats.get("images", 0) + 1
        for key, value in (
            ("full_pass_s", full_pass_s),
            ("full_thing_extract_s", full_thing_extract_s),
            ("router_s", router_s),
            ("tile_forward_s", tile_forward_s),
            ("merge_s", merge_s),
            ("compose_s", compose_s),
            ("postprocess_s", time.perf_counter() - postprocess_started),
        ):
            timing_stats[key] = timing_stats.get(key, 0.0) + value
    return result


def id_to_rgb(ids: np.ndarray) -> np.ndarray:
    ids = ids.astype(np.int64)
    return np.stack(
        (ids % 256, (ids // 256) % 256, (ids // 65536) % 256), axis=-1
    ).astype(np.uint8)


def _images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        for p in sorted(path.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True, help="image or flat image directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--infer-size", type=int, default=None)
    p.add_argument(
        "--tile-config",
        default=None,
        help="tile-expert config; defaults to --config when --tile-ckpt is set",
    )
    p.add_argument(
        "--tile-ckpt",
        default=None,
        help="expert checkpoint used only for tile raw-query forwards",
    )
    p.add_argument("--tile-infer-size", type=int, default=None)
    p.add_argument(
        "--tile-share-encoder",
        action="store_true",
        help="share main/tile encoder only after exact weight verification; "
        "intended for a heads-only expert",
    )
    p.add_argument(
        "--mode", choices=["full-only", "tile-only", "full+tile"], default="full-only"
    )
    p.add_argument("--thing-classes", type=int, nargs="+", default=[0, 1])
    p.add_argument(
        "--tile-size", type=int, nargs=2, default=[1120, 1120], metavar=("H", "W")
    )
    p.add_argument("--tile-overlap", type=float, default=0.2)
    p.add_argument(
        "--score-threshold",
        type=float,
        nargs="+",
        default=[0.1],
        help="tile-branch score threshold. One value applies to every thing "
        "class; pass one per --thing-classes entry, in the same order, to "
        "threshold them asymmetrically -- ship and obstacle do not want "
        "the same operating point",
    )
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--local-nms-iou", type=float, default=0.7)
    p.add_argument("--merge-iou", type=float, default=0.55)
    p.add_argument("--merge-mode", choices=["union", "best"], default="union")
    p.add_argument("--merge-metric", choices=["iou", "iomin"], default="iou")
    p.add_argument("--min-visible-fraction", type=float, default=0.25)
    p.add_argument(
        "--ownership-max-tile-area",
        type=float,
        default=None,
        help="enable conflict-aware ownership only for tile-only merged masks "
        "at or below this area (off by default)",
    )
    p.add_argument(
        "--ownership-min-tile-score",
        type=float,
        default=0.97,
        help="minimum tile score for conflict-aware ownership",
    )
    p.add_argument(
        "--ownership-min-full-tile-area-ratio",
        type=float,
        default=4.0,
        help="minimum conflicting full-mask area divided by tile-mask area",
    )
    p.add_argument(
        "--tile-max-area",
        type=float,
        default=None,
        help="drop tile candidates larger than this pixel area (off by default); "
        "9216 == 96^2 keeps the tile branch to COCO small+medium targets",
    )
    p.add_argument(
        "--sea-min-water",
        type=float,
        default=None,
        help="drop tile candidates whose surrounding band is less than this "
        "fraction water, using the tracker's sea prior (off by default)",
    )
    p.add_argument(
        "--sea-class-id", type=int, default=2, help="internal seasurface class id"
    )
    p.add_argument("--sea-margin-frac", type=float, default=0.5)
    p.add_argument(
        "--gate-stage",
        choices=["before", "after"],
        default="before",
        help="apply the tile gates to raw per-tile candidates ('before') or to "
        "the cross-tile merge result ('after')",
    )
    p.add_argument(
        "--protect-full-above-area",
        type=float,
        nargs="+",
        default=None,
        help="do not let tile fragments union into a full-pass mask already at "
        "least this large (off by default). Union repairs crop-edge "
        "truncation on small targets, but a large ship spans several tiles "
        "and uniting slices only bloats a mask the full pass had right. One "
        "value applies to every thing class; pass one per --thing-classes "
        "entry, in the same order, to set it asymmetrically -- the union "
        "helps obstacle masks (SQ 83.78 united vs 82.97 protected) and "
        "hurts ship's (large fn_poor_mask 60 vs 38), so 0 disables it for "
        "a class and a huge value protects nothing",
    )
    p.add_argument(
        "--tile-router",
        choices=["query-thing", "query-small", "query-uncertainty"],
        default=None,
        help="route a bounded tile subset from full-pass query evidence (off by default)",
    )
    p.add_argument(
        "--tile-budget",
        type=int,
        default=None,
        help="maximum tiles per image when --tile-router is enabled",
    )
    p.add_argument("--router-top-fraction", type=float, default=0.01)
    p.add_argument("--router-min-query-score", type=float, default=0.05)
    p.add_argument(
        "--router-resolution", choices=["fullres", "native"], default="fullres"
    )
    p.add_argument(
        "--router-min-tile-score",
        type=float,
        default=None,
        help="adaptive stop: skip this and later tiles when the next evidence "
        "score is below the threshold (off by default)",
    )
    p.add_argument(
        "--category-id-offset",
        type=int,
        default=1,
        help="add to internal 0-based class ids in JSON (COCO-style sea ids use 1)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.tile_config is not None and args.tile_ckpt is None:
        raise SystemExit("--tile-config requires --tile-ckpt")
    if args.tile_share_encoder and args.tile_ckpt is None:
        raise SystemExit("--tile-share-encoder requires --tile-ckpt")
    if args.tile_ckpt is not None and args.mode not in {"tile-only", "full+tile"}:
        raise SystemExit("--tile-ckpt requires --mode tile-only or full+tile")
    device = args.device if torch.cuda.is_available() else "cpu"
    model = build_inference_model(args.config, args.ckpt, device, args.infer_size)
    tile_model = None
    if args.tile_ckpt is not None:
        tile_model = build_inference_model(
            args.tile_config or args.config,
            args.tile_ckpt,
            "cpu" if args.tile_share_encoder else device,
            args.tile_infer_size,
            require_complete_network=True,
        )
        if args.tile_share_encoder:
            tile_model = share_encoder_if_identical(model, tile_model).to(device).eval()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = args.score_threshold
    if len(thresholds) == 1:
        thresholds = thresholds * len(args.thing_classes)
    if len(thresholds) != len(args.thing_classes):
        raise SystemExit(
            f"--score-threshold takes 1 or {len(args.thing_classes)} values "
            f"(one per --thing-classes entry), got {len(thresholds)}"
        )
    thresholds = dict(zip(args.thing_classes, thresholds))
    protect = _per_class_arg(
        args.protect_full_above_area, args.thing_classes, "--protect-full-above-area"
    )
    gate_stats: Dict[str, int] = {}
    for path in _images(Path(args.input)):
        arr = np.asarray(Image.open(path).convert("RGB"))
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        pan, info = infer_tiled_panoptic(
            model,
            tensor,
            device,
            args.mode,
            args.thing_classes,
            tile_model=tile_model,
            score_thresholds=thresholds,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            mask_threshold=args.mask_threshold,
            local_nms_iou=args.local_nms_iou,
            merge_iou=args.merge_iou,
            merge_mode=args.merge_mode,
            merge_metric=args.merge_metric,
            min_visible_fraction=args.min_visible_fraction,
            ownership_max_tile_area=args.ownership_max_tile_area,
            ownership_min_tile_score=args.ownership_min_tile_score,
            ownership_min_full_tile_area_ratio=(
                args.ownership_min_full_tile_area_ratio
            ),
            tile_max_area=args.tile_max_area,
            sea_min_water=args.sea_min_water,
            sea_class_id=args.sea_class_id,
            sea_margin_frac=args.sea_margin_frac,
            gate_stage=args.gate_stage,
            protect_full_above_area=protect,
            tile_router=args.tile_router,
            tile_budget=args.tile_budget,
            router_top_fraction=args.router_top_fraction,
            router_min_query_score=args.router_min_query_score,
            router_resolution=args.router_resolution,
            router_min_tile_score=args.router_min_tile_score,
            stats=gate_stats,
        )
        Image.fromarray(id_to_rgb(pan)).save(out / f"{path.stem}.png")
        exported_info = [
            {**segment, "category_id": segment["category_id"] + args.category_id_offset}
            for segment in info
        ]
        output = {"file_name": path.name, "segments_info": exported_info}
        if args.tile_ckpt is not None:
            output["tile_model"] = {
                "config": args.tile_config or args.config,
                "checkpoint": args.tile_ckpt,
                "network_input": list(tile_model.img_size),
                "shared_encoder": bool(args.tile_share_encoder),
            }
        if args.ownership_max_tile_area is not None:
            output["conflict_aware_tile_ownership"] = {
                "max_tile_area": args.ownership_max_tile_area,
                "min_tile_score": args.ownership_min_tile_score,
                "min_full_tile_area_ratio": (args.ownership_min_full_tile_area_ratio),
            }
        (out / f"{path.stem}.json").write_text(
            json.dumps(output, indent=2),
            encoding="utf-8",
        )
        print(f"[tiled_panoptic] {path.name}: {len(info)} segments")
    if gate_stats:
        print(f"[tiled_panoptic] gate stats: {gate_stats}")


if __name__ == "__main__":
    main()
