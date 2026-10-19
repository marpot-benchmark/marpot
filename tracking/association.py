
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from tracking import maskutil
from tracking.structures import Detection, Track

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_iou_matrix(dets: Sequence[Detection], preds: Sequence[np.ndarray]) -> np.ndarray:
    m = np.zeros((len(dets), len(preds)), dtype=np.float32)
    for i, d in enumerate(dets):
        for j, p in enumerate(preds):
            m[i, j] = _bbox_iou(d.bbox, p)
    return m


def mask_iou_matrix(dets: Sequence[Detection], tracks: Sequence[Track]) -> Optional[np.ndarray]:
    if not dets or not tracks:
        return np.zeros((len(dets), len(tracks)), dtype=np.float32)
    det_rles = [d.mask_rle for d in dets]
    trk_rles = [t.last_detection.mask_rle for t in tracks]
    if any(r is None for r in det_rles) or any(r is None for r in trk_rles):
        return None
    try:
        return maskutil.iou_matrix(det_rles, trk_rles)
    except Exception:
        return None


def center_dist_matrix(
    dets: Sequence[Detection], pred_centroids: Sequence[np.ndarray], scales: Sequence[float]
) -> np.ndarray:
    m = np.zeros((len(dets), len(pred_centroids)), dtype=np.float32)
    for i, d in enumerate(dets):
        for j, c in enumerate(pred_centroids):
            scale = max(scales[j], 1.0)
            m[i, j] = float(np.linalg.norm(d.centroid - c) / scale)
    return m


def area_change_matrix(dets: Sequence[Detection], tracks: Sequence[Track]) -> np.ndarray:
    m = np.ones((len(dets), len(tracks)), dtype=np.float32)
    for i, d in enumerate(dets):
        for j, t in enumerate(tracks):
            ta = max(t.last_detection.area, 1.0)
            ratio = max(d.area, 1.0) / ta
            m[i, j] = max(ratio, 1.0 / ratio)
    return m


def cosine_matrix(
    dets: Sequence[Detection], tracks: Sequence[Track], source: str
) -> Optional[np.ndarray]:
    if not dets or not tracks:
        return np.zeros((len(dets), len(tracks)), dtype=np.float32)
    det_feats = [d.appearance(source) for d in dets]
    if any(f is None for f in det_feats):
        return None
    trk_feats = [getattr(t, "appearance_protos", {}).get(source) for t in tracks]
    if any(f is None for f in trk_feats):
        return None
    D = np.stack([_l2(f) for f in det_feats])
    T = np.stack([_l2(f) for f in trk_feats])
    if D.shape[1] != T.shape[1]:
        return None
    sim = D @ T.T
    cost = np.nan_to_num(1.0 - sim, nan=1.0, posinf=1.0, neginf=1.0)
    return cost.astype(np.float32)


def query_cosine_matrix(
    dets: Sequence[Detection], tracks: Sequence[Track]
) -> Optional[np.ndarray]:
    return cosine_matrix(dets, tracks, "query_embed")


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(np.asarray(v, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def linear_assignment(
    cost: np.ndarray, max_cost: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    if linear_sum_assignment is None:
        raise ImportError("scipy is required for Hungarian matching")

    if not np.isfinite(cost).all():
        cost = np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)

    row_ind, col_ind = linear_sum_assignment(cost)
    matches, used_r, used_c = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            used_r.add(int(r))
            used_c.add(int(c))
    unmatched_rows = [r for r in range(n_rows) if r not in used_r]
    unmatched_cols = [c for c in range(n_cols) if c not in used_c]
    return matches, unmatched_rows, unmatched_cols
