
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp.autocast_mode import autocast

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tracking import maskutil


def _import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def build_inference_model(
    config_path: str,
    ckpt_path: str,
    device: str = "cuda:0",
    infer_size: Optional[int] = None,
    return_query_embed: bool = False,
    return_mask_pooled: bool = False,
    reid_enabled: bool = False,
    require_complete_network: bool = False,
    require_reid_head: bool = False,
):
    if require_reid_head and not reid_enabled:
        raise ValueError("require_reid_head requires reid_enabled=True")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_init = config["data"].get("init_args", {})
    num_classes = data_init.get("num_classes", 4)
    stuff_classes = data_init.get("stuff_classes", [])

    if infer_size is not None:
        img_size = (int(infer_size), int(infer_size))
    else:
        img_size = tuple(data_init.get("img_size", (640, 640)))

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_cls = _import_class(encoder_cfg["class_path"])
    encoder_init = dict(encoder_cfg.get("init_args", {}))
    encoder_init.setdefault("ckpt_path", ckpt_path)
    encoder = encoder_cls(img_size=img_size, **encoder_init)

    network_cfg = config["model"]["init_args"]["network"]
    network_cls = _import_class(network_cfg["class_path"])
    _explicit = {"encoder", "return_query_embed", "return_mask_pooled", "reid_enabled"}
    network_kwargs = {
        k: v for k, v in network_cfg.get("init_args", {}).items() if k not in _explicit
    }
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=num_classes,
        encoder=encoder,
        return_query_embed=return_query_embed,
        return_mask_pooled=return_mask_pooled,
        reid_enabled=reid_enabled,
        **network_kwargs,
    )

    lit_cls = _import_class(config["model"]["class_path"])
    model_kwargs = {
        k: v
        for k, v in config["model"]["init_args"].items()
        if k not in ("network", "ckpt_path", "delta_weights")
    }
    model_kwargs["stuff_classes"] = stuff_classes
    model_kwargs.setdefault("load_ckpt_reid_head", False)

    model = lit_cls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    )

    _load_state_dict(
        model,
        ckpt_path,
        device,
        require_complete_network=require_complete_network,
        require_reid_head=require_reid_head,
    )
    return model.eval().to(device)


def share_encoder_if_identical(main_model, tile_model):
    main_encoder = main_model.network.encoder
    tile_encoder = tile_model.network.encoder
    if type(main_encoder) is not type(tile_encoder):
        raise ValueError("main and tile encoders have different module types")
    main_state = main_encoder.state_dict()
    tile_state = tile_encoder.state_dict()
    if main_state.keys() != tile_state.keys():
        raise ValueError("main and tile encoders have different state-dict keys")
    for name, tile_value in tile_state.items():
        main_value = main_state[name]
        if (
            main_value.shape != tile_value.shape
            or main_value.dtype != tile_value.dtype
            or not torch.equal(main_value.detach().cpu(), tile_value.detach().cpu())
        ):
            raise ValueError(
                "--tile-share-encoder requires exactly identical encoder weights; "
                f"first mismatch: {name}"
            )
    tile_model.network.encoder = main_encoder
    return tile_model


def _resize_pos_embed(pos_embed: torch.Tensor, target_n: int) -> torch.Tensor:
    n_old = pos_embed.shape[1]
    g_old, g_new = int(round(n_old**0.5)), int(round(target_n**0.5))
    if g_old * g_old != n_old or g_new * g_new != target_n:
        raise ValueError(
            f"Cannot resize pos_embed from {n_old} to {target_n}: non-square grid"
        )
    c = pos_embed.shape[2]
    grid = pos_embed.reshape(1, g_old, g_old, c).permute(0, 3, 1, 2).float()
    grid = F.interpolate(grid, size=(g_new, g_new), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(1, target_n, c).to(pos_embed.dtype)


def _load_state_dict(
    model,
    ckpt_path: str,
    device: str,
    require_complete_network: bool = False,
    require_reid_head: bool = False,
):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state = {k.replace("._orig_mod", ""): v for k, v in state.items()}

    model_state = model.state_dict()
    for k, v in list(state.items()):
        if (
            k.endswith("pos_embed")
            and k in model_state
            and model_state[k].shape != v.shape
        ):
            try:
                state[k] = _resize_pos_embed(v, model_state[k].shape[1])
                print(
                    f"[eomt_infer] resized {k}: {tuple(v.shape)} -> {tuple(state[k].shape)}"
                )
            except ValueError as e:
                print(f"[eomt_infer] WARNING {e}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if require_reid_head:
        missing_reid = [k for k in missing if "reid_head" in k]
        if missing_reid:
            raise ValueError(
                "--save_reid requires checkpoint ReID weights; missing keys: "
                f"{missing_reid}"
            )
    missing = [k for k in missing if "reid_head" not in k]
    if require_complete_network:
        missing_network = [k for k in missing if k.startswith("network.")]
        if missing_network:
            preview = missing_network[:8]
            suffix = "..." if len(missing_network) > len(preview) else ""
            raise ValueError(
                "incomplete tile-expert checkpoint; missing network keys: "
                f"{preview}{suffix}"
            )
    if missing:
        print(
            f"[eomt_infer] WARNING missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    if unexpected:
        unexpected = [
            k for k in unexpected if not k.startswith(("criterion.", "metrics."))
        ]
        if require_complete_network:
            unexpected_network = [
                k
                for k in unexpected
                if k.startswith("network.") and "reid_head" not in k
            ]
            if unexpected_network:
                preview = unexpected_network[:8]
                suffix = "..." if len(unexpected_network) > len(preview) else ""
                raise ValueError(
                    "incompatible tile-expert checkpoint; unexpected network keys: "
                    f"{preview}{suffix}"
                )
        if unexpected:
            print(f"[eomt_infer] WARNING unexpected keys: {unexpected[:8]}")


def _encode_rle(mask_np: np.ndarray) -> Dict:
    return maskutil.encode(mask_np)


def _mask_nms(rles: List[Dict], scores: np.ndarray, iou_thr: float) -> List[int]:
    return maskutil.nms(rles, scores, iou_thr)


@torch.no_grad()
def infer_raw_query_detections(
    model,
    img: torch.Tensor,
    device: str,
    thing_classes: Sequence[int],
    class_names: Dict[int, str],
    score_thr_by_class: Dict[int, float],
    mask_bin_thr: float = 0.5,
    mask_nms_iou: float = 0.7,
    prefilter_score: float = 0.05,
    max_candidates: int = 200,
    want_query_embed: bool = False,
    want_mask_pooled: bool = False,
    want_reid: bool = False,
    xy_offset: Sequence[float] = (0.0, 0.0),
    canvas_size: Optional[Sequence[int]] = None,
    want_tracking_score: bool = False,
    raw_trace_sink: Optional[List[Dict]] = None,
) -> List[Dict]:
    img = img.to(device)
    orig_h, orig_w = int(img.shape[-2]), int(img.shape[-1])

    transformed = model.resize_and_pad_imgs_instance_panoptic([img])
    with autocast(
        dtype=torch.float16, device_type="cuda" if "cuda" in device else "cpu"
    ):
        outputs = model(transformed)
    mask_logits = outputs[0][-1]
    class_logits = outputs[1][-1]
    aux = outputs[2][-1] if len(outputs) > 2 else None

    probs = class_logits.softmax(dim=-1)[0]
    thing_idx = torch.tensor(list(thing_classes), device=probs.device)
    thing_probs = probs[:, thing_idx]
    best_score, best_local = thing_probs.max(dim=-1)
    best_class = thing_idx[best_local]

    keep = best_score >= (0.0 if raw_trace_sink is not None else prefilter_score)
    if keep.sum() == 0:
        return []
    cand = keep.nonzero(as_tuple=False).squeeze(1)
    if cand.numel() > max_candidates:
        topv, topi = best_score[cand].topk(max_candidates)
        cand = cand[topi]

    ml = mask_logits[:, cand]
    ml = F.interpolate(ml.float(), model.img_size, mode="bilinear")
    ml = model.revert_resize_and_pad_logits_instance_panoptic(ml, [(orig_h, orig_w)])[0]
    prob = ml.sigmoid()
    masks = prob > mask_bin_thr

    flat = masks.flatten(1).float()
    areas = flat.sum(1)
    mask_score = (prob.flatten(1) * flat).sum(1) / (areas + 1e-6)
    final_score = best_score[cand] * mask_score

    masks_np = masks.cpu().numpy()
    cls_np = best_class[cand].cpu().numpy()
    score_np = final_score.cpu().numpy()
    cand_np = cand.cpu().numpy()
    areas_np = areas.cpu().numpy()

    ox, oy = float(xy_offset[0]), float(xy_offset[1])
    if canvas_size is not None:
        H, W = int(canvas_size[0]), int(canvas_size[1])
    else:
        H, W = orig_h, orig_w

    q_embed = q_pool = q_reid = None
    if aux is not None:
        if want_query_embed and "query_embed" in aux:
            q_embed = aux["query_embed"][0][cand].float().cpu().numpy()
        if want_mask_pooled and "mask_pooled" in aux:
            q_pool = aux["mask_pooled"][0][cand].float().cpu().numpy()
        if want_reid and "reid" in aux:
            q_reid = aux["reid"][0][cand].float().cpu().numpy()

    if raw_trace_sink is not None:
        raw_trace_sink.clear()
        for k in range(len(cls_np)):
            if areas_np[k] < 1:
                continue
            raw_trace_sink.append({
                "query_index": int(cand_np[k]), "class_id": int(cls_np[k]),
                "class_score": float(best_score[cand[k]].item()),
                "score": float(score_np[k]), "mask_rle_local": _encode_rle(masks_np[k]),
                "xy_offset": [int(ox), int(oy)], "canvas_size": [int(H), int(W)],
                "score_keep": bool(best_score[cand[k]] >= prefilter_score and score_np[k] >= score_thr_by_class.get(int(cls_np[k]), 0.5)),
                "nms_keep": False,
            })
    kept_local = []
    rles, rle_scores = [], []
    for k in range(len(cls_np)):
        cid = int(cls_np[k])
        if areas_np[k] < 1:
            continue
        if best_score[cand[k]] < prefilter_score or score_np[k] < score_thr_by_class.get(cid, 0.5):
            continue
        if canvas_size is not None or ox or oy:
            canvas = np.zeros((H, W), dtype=np.uint8)
            m = masks_np[k]
            y0, x0 = int(round(oy)), int(round(ox))
            h_m, w_m = m.shape
            canvas[y0 : y0 + h_m, x0 : x0 + w_m] = m
            rle = _encode_rle(canvas)
        else:
            rle = _encode_rle(masks_np[k])
        rles.append(rle)
        rle_scores.append(score_np[k])
        kept_local.append(k)

    keep_idx = _mask_nms(rles, np.array(rle_scores), mask_nms_iou)

    dets = []
    for ki in keep_idx:
        k = kept_local[ki]
        rle = rles[ki]
        x, y, w, h = maskutil.to_bbox(rle)
        det = {
            "class_id": int(cls_np[k]),
            "class_name": class_names.get(int(cls_np[k]), str(int(cls_np[k]))),
            "score": float(score_np[k]),
            "bbox": [x, y, x + w, y + h],
            "centroid": [x + w / 2, y + h / 2],
            "area": float(maskutil.area(rle)),
            "aspect_ratio": float(w / max(h, 1e-3)),
            "mask_rle": rle,
            "query_index": int(cand_np[k]),
        }
        if q_embed is not None:
            det["query_embed"] = [round(float(v), 5) for v in q_embed[k]]
        if q_pool is not None:
            det["mask_pooled"] = [round(float(v), 5) for v in q_pool[k]]
        if q_reid is not None:
            det["reid"] = [round(float(v), 5) for v in q_reid[k]]
        if want_tracking_score:
            det["tracking_score"] = float(best_score[cand[k]].item())
        dets.append(det)
    if raw_trace_sink is not None:
        kept_queries = {d["query_index"] for d in dets}
        for item in raw_trace_sink:
            item["nms_keep"] = item["query_index"] in kept_queries
    return dets


def classify_query_recoverability(
    best_mask_iou_any: float,
    best_same_class_iou: float,
    best_matching_final_score: float,
    score_threshold: float,
    best_post_nms_iou: float,
) -> str:
    if best_mask_iou_any < 0.1:
        return "no_mask_response"
    if best_same_class_iou < 0.1:
        return "classification_mismatch"
    if best_same_class_iou <= 0.5:
        return "poor_mask"
    if best_matching_final_score < score_threshold:
        return "score_below_threshold"
    if best_post_nms_iou <= 0.5:
        return "local_nms_lost"
    return "post_nms_recoverable"


@torch.no_grad()
def infer_query_recoverability(
    model,
    img: torch.Tensor,
    target_mask: torch.Tensor,
    gt_class_id: int,
    device: str,
    thing_classes: Sequence[int] = (0, 1),
    class_names: Optional[Dict[int, str]] = None,
    score_thr_by_class: Optional[Dict[int, float]] = None,
    mask_bin_thr: float = 0.5,
    mask_nms_iou: float = 0.7,
    xy_offset: Sequence[float] = (0.0, 0.0),
    canvas_size: Optional[Sequence[int]] = None,
) -> Tuple[Dict, List[Dict]]:
    class_names = class_names or {int(c): str(c) for c in thing_classes}
    score_thr_by_class = score_thr_by_class or {int(c): 0.95 for c in thing_classes}
    img = img.to(device)
    target_mask = torch.as_tensor(target_mask, device=device).bool()
    orig_h, orig_w = int(img.shape[-2]), int(img.shape[-1])
    if tuple(target_mask.shape) != (orig_h, orig_w):
        raise ValueError("target_mask must match the crop image size")

    transformed = model.resize_and_pad_imgs_instance_panoptic([img])
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with autocast(
        dtype=torch.float16, device_type=device_type, enabled=device_type == "cuda"
    ):
        outputs = model(transformed)

    mask_logits = F.interpolate(outputs[0][-1].float(), model.img_size, mode="bilinear")
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
        mask_logits, [(orig_h, orig_w)]
    )[0]
    mask_prob = mask_logits.sigmoid()
    masks = mask_prob > mask_bin_thr
    flat = masks.flatten(1)
    areas = flat.sum(1)
    target_area = target_mask.sum().clamp_min(1)
    intersections = (masks & target_mask[None]).flatten(1).sum(1)
    unions = areas + target_area - intersections
    ious = intersections.float() / unions.clamp_min(1).float()
    mask_scores = (mask_prob.flatten(1) * flat.float()).sum(1) / areas.clamp_min(
        1
    ).float()

    class_prob = outputs[1][-1].float().softmax(dim=-1)[0]
    thing_index = torch.as_tensor(list(thing_classes), device=device, dtype=torch.long)
    thing_scores, thing_local = class_prob.index_select(-1, thing_index).max(dim=-1)
    best_classes = thing_index[thing_local]
    final_scores = thing_scores * mask_scores
    valid = areas > 0
    same_class = valid & best_classes.eq(int(gt_class_id))
    matching = same_class & ious.gt(0.5)

    def best_value(values: torch.Tensor, keep: torch.Tensor) -> float:
        return float(values[keep].max().item()) if bool(keep.any()) else 0.0

    best_any_index = int(torch.argmax(ious).item())
    best_mask_iou_any = float(ious[best_any_index].item())
    best_same_class_iou = best_value(ious, same_class)
    best_same_index = (
        int(torch.where(same_class, ious, torch.full_like(ious, -1)).argmax().item())
        if bool(same_class.any())
        else -1
    )
    best_matching_final_score = best_value(final_scores, matching)
    gt_threshold = float(score_thr_by_class.get(int(gt_class_id), 0.95))

    per_query_threshold = torch.full_like(final_scores, float("inf"))
    for class_id, threshold in score_thr_by_class.items():
        per_query_threshold[best_classes.eq(int(class_id))] = float(threshold)
    passing = valid & final_scores.ge(per_query_threshold)
    passing_indices = torch.nonzero(passing, as_tuple=False).flatten().tolist()

    masks_np = masks[passing].cpu().numpy()
    passing_classes = best_classes[passing].cpu().numpy()
    passing_scores = final_scores[passing].cpu().numpy()
    ox, oy = float(xy_offset[0]), float(xy_offset[1])
    canvas_h, canvas_w = (
        (int(canvas_size[0]), int(canvas_size[1]))
        if canvas_size is not None
        else (orig_h, orig_w)
    )
    rles, detections = [], []
    for local_index, query_index in enumerate(passing_indices):
        mask_np = masks_np[local_index]
        if canvas_size is not None or ox or oy:
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            y0, x0 = int(round(oy)), int(round(ox))
            canvas[y0 : y0 + orig_h, x0 : x0 + orig_w] = mask_np
            rle = _encode_rle(canvas)
        else:
            rle = _encode_rle(mask_np)
        rles.append(rle)
        x, y, box_w, box_h = maskutil.to_bbox(rle)
        detections.append(
            {
                "class_id": int(passing_classes[local_index]),
                "class_name": class_names.get(
                    int(passing_classes[local_index]),
                    str(int(passing_classes[local_index])),
                ),
                "score": float(passing_scores[local_index]),
                "bbox": [x, y, x + box_w, y + box_h],
                "centroid": [x + box_w / 2, y + box_h / 2],
                "area": float(maskutil.area(rle)),
                "aspect_ratio": float(box_w / max(box_h, 1e-3)),
                "mask_rle": rle,
                "query_index": int(query_index),
            }
        )

    keep_local = (
        _mask_nms(rles, np.asarray(passing_scores), mask_nms_iou) if rles else []
    )
    kept_detections = [detections[index] for index in keep_local]
    kept_queries = [passing_indices[index] for index in keep_local]
    post_nms_same = [
        query_index
        for query_index in kept_queries
        if int(best_classes[query_index]) == int(gt_class_id)
    ]
    best_post_nms_iou = (
        max(float(ious[query_index]) for query_index in post_nms_same)
        if post_nms_same
        else 0.0
    )
    best_passing_iou = best_value(ious, same_class & passing)
    premerge_stage = classify_query_recoverability(
        best_mask_iou_any,
        best_same_class_iou,
        best_matching_final_score,
        gt_threshold,
        best_post_nms_iou,
    )
    diagnostic = {
        "best_mask_iou_any": best_mask_iou_any,
        "best_mask_any_class": int(best_classes[best_any_index]),
        "best_mask_any_thing_score": float(thing_scores[best_any_index]),
        "best_same_class_iou": best_same_class_iou,
        "best_same_class_final_score": (
            float(final_scores[best_same_index].item()) if best_same_index >= 0 else 0.0
        ),
        "best_matching_final_score": best_matching_final_score,
        "best_passing_iou": best_passing_iou,
        "best_post_nms_iou": best_post_nms_iou,
        "score_threshold": gt_threshold,
        "passing_queries": len(passing_indices),
        "kept_after_nms": len(kept_queries),
        "premerge_stage": premerge_stage,
    }
    return diagnostic, kept_detections


def query_routing_evidence(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    thing_classes: Sequence[int] = (0, 1),
    mode: str = "query-thing",
    min_query_score: float = 0.05,
    options: Optional[Dict] = None,
    retained_queries: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mode not in {"query-thing", "query-small", "query-uncertainty"}:
        raise ValueError(f"unsupported query routing mode: {mode}")
    if not 0 <= min_query_score <= 1:
        raise ValueError("min_query_score must be in [0, 1]")

    probs = class_logits.float().softmax(dim=-1)
    thing_index = torch.as_tensor(
        list(thing_classes), device=probs.device, dtype=torch.long
    )
    thing_score = probs.index_select(-1, thing_index).amax(dim=-1)
    mask_evidence = (2.0 * mask_logits.float().sigmoid() - 1.0).clamp_min(0.0)
    area = (mask_evidence > 0).flatten(2).sum(dim=-1).clamp_min(1).float()
    valid = (thing_score >= min_query_score) & (mask_evidence.amax(dim=(-2, -1)) > 0)
    options = options or {}
    query_set = options.get("query_set", "all")
    if query_set == "retained":
        if retained_queries is None:
            raise ValueError("retained query evidence requires emitted query ids")
        valid = valid & retained_queries
    elif query_set == "score_qualified":
        scores, labels = probs.max(-1)
        is_thing = (labels[..., None] == thing_index).any(-1)
        valid = valid & is_thing & (scores > options.get("full_threshold", 0.8))
    elif query_set != "all":
        raise ValueError("unknown query_set")
    weight = thing_score * valid
    gamma = float(options.get("gamma", 0.5 if mode == "query-small" else 0.0))
    if not __import__("math").isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    weight = weight / area.pow(gamma)
    if options.get("semantic_prob", False):
        evidence = (mask_logits.float().sigmoid() * probs.index_select(-1, thing_index).sum(-1)[..., None, None]).sum(1, keepdim=True)
    else:
        evidence = (mask_evidence * weight[..., None, None]).amax(dim=1, keepdim=True)
    if mode == "query-uncertainty":
        eps = torch.finfo(evidence.dtype).eps
        probability = evidence.clamp(min=eps, max=1.0 - eps)
        entropy = -probability * probability.log()
        entropy -= (1.0 - probability) * (1.0 - probability).log()
        evidence = torch.where(evidence > 0, entropy, torch.zeros_like(entropy))
    return evidence


def panoptic_thing_query_metadata(
    pred: torch.Tensor,
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    aux: Optional[Dict],
    thing_classes: Sequence[int],
    mask_thresh: float,
) -> Dict[int, Dict]:
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError("panoptic prediction must have shape [H, W, 2]")
    if mask_logits.ndim != 3:
        raise ValueError("panoptic mask logits must have shape [Q, H, W]")
    if class_logits.ndim != 2:
        raise ValueError("panoptic class logits must have shape [Q, C+1]")

    scores, classes = class_logits.softmax(dim=-1).max(dim=-1)
    keep = classes.ne(class_logits.shape[-1] - 1) & (scores > float(mask_thresh))
    kept_queries = keep.nonzero(as_tuple=False).squeeze(1)
    metadata: Dict[int, Dict] = {}
    if kept_queries.numel() == 0:
        return metadata

    thing_set = {int(class_id) for class_id in thing_classes}
    segment_map = pred[..., 1]
    class_map = pred[..., 0]
    for segment_id_tensor in torch.unique(segment_map):
        segment_id = int(segment_id_tensor.item())
        if segment_id < 0:
            continue
        region = segment_map == segment_id
        class_id = int(class_map[region][0].item())
        if class_id not in thing_set:
            continue

        y, x = region.nonzero(as_tuple=False)[0]
        local_winner = torch.argmax(
            scores[kept_queries] * mask_logits[kept_queries, y, x].sigmoid()
        )
        query_index = int(kept_queries[local_winner].item())
        query_class = int(classes[query_index].item())
        if query_class != class_id:
            raise RuntimeError(
                "panoptic segment/query class mismatch: "
                f"segment {segment_id} has class {class_id}, query "
                f"{query_index} has class {query_class}"
            )

        item = {
            "query_index": query_index,
            "tracking_score": float(scores[query_index].item()),
        }
        if aux is not None:
            for feature in ["query_embed", "mask_pooled"]:
                if feature in aux:
                    item[feature] = aux[feature][0, query_index].float().cpu().tolist()
        if aux is not None and "reid" in aux:
            reid = aux["reid"][0, query_index].float().cpu().numpy()
            item["reid"] = [round(float(value), 5) for value in reid]
        metadata[segment_id] = item
    return metadata


@torch.no_grad()
def infer_panoptic_map(
    model,
    img: torch.Tensor,
    device: str,
    routing_mode: Optional[str] = None,
    routing_thing_classes: Sequence[int] = (0, 1),
    routing_min_query_score: float = 0.05,
    routing_resolution: str = "fullres",
    thing_query_metadata_sink: Optional[Dict[int, Dict]] = None,
):
    img = img.to(device)
    orig_h, orig_w = int(img.shape[-2]), int(img.shape[-1])
    transformed = model.resize_and_pad_imgs_instance_panoptic([img])
    with autocast(
        dtype=torch.float16, device_type="cuda" if "cuda" in device else "cpu"
    ):
        outputs = model(transformed)
    if routing_resolution not in {"fullres", "native"}:
        raise ValueError("routing_resolution must be 'fullres' or 'native'")
    mask_logits = F.interpolate(outputs[0][-1].float(), model.img_size, mode="bilinear")
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
        mask_logits, [(orig_h, orig_w)]
    )
    pred = model.to_per_pixel_preds_panoptic(
        mask_logits,
        outputs[1][-1],
        model.stuff_classes,
        model.mask_thresh,
        model.overlap_thresh,
    )[0]
    options = getattr(model, "supplement_options", {})
    retained = None
    if options.get("query_set") == "retained":
        metadata = panoptic_thing_query_metadata(pred, mask_logits[0], outputs[1][-1][0], None, routing_thing_classes, model.mask_thresh)
        retained = torch.zeros(outputs[1][-1].shape[:2], dtype=torch.bool, device=outputs[1][-1].device)
        for item in metadata.values():
            retained[0, item["query_index"]] = True
    routing_map = None
    if routing_mode is not None:
        routing_evidence = query_routing_evidence(
            outputs[0][-1],
            outputs[1][-1],
            routing_thing_classes,
            routing_mode,
            routing_min_query_score,
            options=options, retained_queries=retained,
        )
        if routing_resolution == "fullres":
            routing_map = F.interpolate(
                routing_evidence, model.img_size, mode="bilinear"
            )
            routing_map = (
                model.revert_resize_and_pad_logits_instance_panoptic(
                    routing_map, [(orig_h, orig_w)]
                )[0][0]
                .cpu()
                .numpy()
            )
        else:
            routing_map = {
                "evidence": routing_evidence[0, 0],
                "image_size": (orig_h, orig_w),
                "scaled_size": tuple(
                    model.scale_img_size_instance_panoptic((orig_h, orig_w))
                ),
                "padded_size": tuple(model.img_size),
            }
    if thing_query_metadata_sink is not None:
        aux = outputs[2][-1] if len(outputs) > 2 and outputs[2] else None
        thing_query_metadata_sink.clear()
        thing_query_metadata_sink.update(
            panoptic_thing_query_metadata(
                pred,
                mask_logits[0],
                outputs[1][-1][0],
                aux,
                routing_thing_classes,
                model.mask_thresh,
            )
        )
    result = (pred[..., 0].cpu().numpy(), pred[..., 1].cpu().numpy())
    return (*result, routing_map) if routing_map is not None else result


@torch.no_grad()
def infer_panoptic_argmax_detections(
    model,
    img: torch.Tensor,
    device: str,
    thing_classes: Sequence[int],
    class_names: Dict[int, str],
    score_thr_by_class: Optional[Dict[int, float]] = None,
) -> List[Dict]:
    img = img.to(device)
    orig_h, orig_w = int(img.shape[-2]), int(img.shape[-1])
    transformed = model.resize_and_pad_imgs_instance_panoptic([img])
    with autocast(
        dtype=torch.float16, device_type="cuda" if "cuda" in device else "cpu"
    ):
        outputs = model(transformed)
    class_logits = outputs[1][-1]
    mask_logits = F.interpolate(outputs[0][-1].float(), model.img_size, mode="bilinear")
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
        mask_logits, [(orig_h, orig_w)]
    )
    preds = model.to_per_pixel_preds_panoptic(
        mask_logits,
        class_logits,
        model.stuff_classes,
        model.mask_thresh,
        model.overlap_thresh,
    )[0]
    cls_map = preds[..., 0].cpu().numpy()
    seg_map_t = preds[..., 1]

    q_scores = class_logits.softmax(dim=-1)[0].max(dim=-1).values
    native = outputs[0][-1][0].sigmoid()
    winner_native = (q_scores[:, None, None] * native).argmax(0)
    h, w = winner_native.shape
    seg_native = F.interpolate(
        seg_map_t[None, None].float(), size=(h, w), mode="nearest"
    )[0, 0].long()
    q_scores_np = q_scores.float().cpu().numpy()
    winner_np = winner_native.cpu().numpy()
    seg_native_np = seg_native.cpu().numpy()
    seg_map = seg_map_t.cpu().numpy()

    if score_thr_by_class is None:
        score_thr_by_class = {}

    dets = []
    for seg_id in np.unique(seg_map):
        if seg_id < 0:
            continue
        m = seg_map == seg_id
        cid = int(cls_map[m].flat[0])
        if cid not in thing_classes:
            continue
        region = seg_native_np == seg_id
        if region.any():
            winners = winner_np[region]
            winner_q = int(np.bincount(winners).argmax())
            score = float(q_scores_np[winner_q])
        else:
            score = float(q_scores_np.max())
        if score < score_thr_by_class.get(cid, 0.0):
            continue
        rle = _encode_rle(m)
        x, y, w_b, h_b = maskutil.to_bbox(rle)
        if w_b * h_b < 1:
            continue
        dets.append(
            {
                "class_id": cid,
                "class_name": class_names.get(cid, str(cid)),
                "score": score,
                "bbox": [x, y, x + w_b, y + h_b],
                "centroid": [x + w_b / 2, y + h_b / 2],
                "area": float(maskutil.area(rle)),
                "aspect_ratio": float(w_b / max(h_b, 1e-3)),
                "mask_rle": rle,
                "query_index": winner_q if region.any() else None,
            }
        )
    return dets


@torch.no_grad()
def infer_seasurface_rle(
    model,
    img: torch.Tensor,
    device: str,
    seasurface_class: int = 2,
) -> Optional[Dict]:
    img = img.to(device)
    orig_h, orig_w = int(img.shape[-2]), int(img.shape[-1])
    transformed = model.resize_and_pad_imgs_instance_panoptic([img])
    with autocast(
        dtype=torch.float16, device_type="cuda" if "cuda" in device else "cpu"
    ):
        outputs = model(transformed)
    mask_logits = F.interpolate(outputs[0][-1].float(), model.img_size, mode="bilinear")
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
        mask_logits, [(orig_h, orig_w)]
    )
    preds = model.to_per_pixel_preds_panoptic(
        mask_logits,
        outputs[1][-1],
        model.stuff_classes,
        model.mask_thresh,
        model.overlap_thresh,
    )[0]
    cls_map = preds[..., 0].cpu().numpy()
    sea = cls_map == seasurface_class
    if sea.sum() == 0:
        return None
    return _encode_rle(sea)
