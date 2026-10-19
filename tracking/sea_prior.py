
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from tracking import maskutil
from tracking.config import TrackerConfig
from tracking.structures import Detection


def _decode(rle) -> Optional[np.ndarray]:
    try:
        return maskutil.decode(rle)
    except Exception:
        return None


def water_proximity(
    bbox: Sequence[float],
    mask: Optional[np.ndarray],
    sea: np.ndarray,
    margin_frac: float,
) -> float:
    H, W = sea.shape
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    mx, my = max(4.0, bw * margin_frac), max(4.0, bh * margin_frac)
    ex1, ey1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    ex2, ey2 = min(W, int(x2 + mx)), min(H, int(y2 + my))
    if ex2 <= ex1 or ey2 <= ey1:
        return 0.0
    sea_crop = sea[ey1:ey2, ex1:ex2]

    if mask is not None and mask.shape == sea.shape:
        neigh = ~mask[ey1:ey2, ex1:ex2]
        denom = int(neigh.sum())
        if denom == 0:
            return 0.0
        return float((sea_crop & neigh).sum() / denom)
    return float(sea_crop.mean())


class SeaPrior:
    def __init__(self, config: TrackerConfig):
        self.cfg = config

    def apply(
        self, detections: List[Detection], seasurface_rle: Optional[dict]
    ) -> List[Detection]:
        if not self.cfg.sea_prior_enabled or seasurface_rle is None:
            return detections
        sea = _decode(seasurface_rle)
        if sea is None:
            return detections

        H, W = sea.shape
        kept: List[Detection] = []
        for d in detections:
            overlap = self._water_proximity(d, sea, H, W, self.cfg.sea_margin_frac)
            if overlap >= self.cfg.sea_overlap_min:
                kept.append(d)
            else:
                d.score *= overlap / max(self.cfg.sea_overlap_min, 1e-6)
                if d.score >= self.cfg.min_score_for_class(d.class_id):
                    kept.append(d)
        return kept

    @staticmethod
    def _water_proximity(
        det: Detection, sea: np.ndarray, H: int, W: int, margin_frac: float
    ) -> float:
        mask = _decode(det.mask_rle) if det.mask_rle is not None else None
        return water_proximity(det.bbox, mask, sea, margin_frac)
