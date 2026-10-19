
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _counts_bytes(rle: Dict) -> bytes:
    c = rle["counts"]
    return c.encode("ascii") if isinstance(c, str) else c


def decode_counts(counts) -> List[int]:
    s = counts.encode("ascii") if isinstance(counts, str) else counts
    cnts: List[int] = []
    m = 0
    p = 0
    n = len(s)
    while p < n:
        x = 0
        k = 0
        more = 1
        while more:
            c = s[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1
    return cnts


def decode(rle: Dict) -> np.ndarray:
    h, w = int(rle["size"][0]), int(rle["size"][1])
    cnts = decode_counts(rle["counts"])
    vals = np.zeros(len(cnts), dtype=np.uint8)
    vals[1::2] = 1
    flat = np.repeat(vals, cnts)
    if flat.size != h * w:
        out = np.zeros(h * w, dtype=bool)
        out[: flat.size] = flat.astype(bool)
        flat = out
    return flat.reshape((h, w), order="F").astype(bool)


def encode(mask: np.ndarray) -> Dict:
    mask = np.asfortranarray(mask.astype(np.uint8))
    try:
        from pycocotools import mask as _m

        rle = _m.encode(mask)
        counts = rle["counts"]
        return {
            "size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts,
        }
    except Exception:
        return _encode_pure(mask)


def _encode_pure(mask: np.ndarray) -> Dict:
    h, w = mask.shape
    flat = mask.reshape(-1, order="F").astype(np.uint8)
    runs: List[int] = []
    prev = 0
    count = 0
    for v in flat:
        if v == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
            prev = v
    runs.append(count)
    return {"size": [int(h), int(w)], "counts": _encode_counts(runs)}


def _encode_counts(cnts: Sequence[int]) -> str:
    s = bytearray()
    for i, value in enumerate(cnts):
        x = int(value)
        if i > 2:
            x -= int(cnts[i - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            s.append(c + 48)
    return s.decode("ascii")


def area(rle: Dict) -> int:
    cnts = decode_counts(rle["counts"])
    return int(sum(cnts[1::2]))


def to_bbox(rle: Dict) -> List[float]:
    m = decode(rle)
    ys, xs = np.where(m)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    return [x1, y1, x2 - x1 + 1.0, y2 - y1 + 1.0]


def iou(rle_a: Dict, rle_b: Dict, ma: Optional[np.ndarray] = None, mb: Optional[np.ndarray] = None) -> float:
    a = decode(rle_a) if ma is None else ma
    b = decode(rle_b) if mb is None else mb
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(rles_a: Sequence[Dict], rles_b: Sequence[Dict]) -> np.ndarray:
    if not rles_a or not rles_b:
        return np.zeros((len(rles_a), len(rles_b)), dtype=np.float32)
    masks_a = [decode(r) for r in rles_a]
    masks_b = [decode(r) for r in rles_b]
    out = np.zeros((len(masks_a), len(masks_b)), dtype=np.float32)
    areas_b = [m.sum() for m in masks_b]
    for i, ma in enumerate(masks_a):
        area_a = ma.sum()
        for j, mb in enumerate(masks_b):
            inter = np.logical_and(ma, mb).sum()
            if inter == 0:
                continue
            union = area_a + areas_b[j] - inter
            out[i, j] = inter / union if union > 0 else 0.0
    return out


def nms(rles: Sequence[Dict], scores: np.ndarray, iou_thr: float, classes: Optional[Sequence[int]] = None) -> List[int]:
    if len(rles) == 0:
        return []
    masks = [decode(r) for r in rles]
    areas = [m.sum() for m in masks]
    order = list(np.argsort(-np.asarray(scores)))
    keep, suppressed = [], set()
    for ii, i in enumerate(order):
        if i in suppressed:
            continue
        keep.append(i)
        for j in order[ii + 1 :]:
            if j in suppressed:
                continue
            if classes is not None and classes[j] != classes[i]:
                continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            if inter == 0:
                continue
            union = areas[i] + areas[j] - inter
            if union > 0 and inter / union > iou_thr:
                suppressed.add(j)
    return keep
