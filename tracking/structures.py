
from __future__ import annotations

import enum
from typing import Dict, List, Optional

import numpy as np


class TrackState(enum.IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


APPEARANCE_SOURCES = ("query_embed", "mask_pooled", "reid")


def bbox_from_mask_rle(rle: Dict) -> Optional[np.ndarray]:
    try:
        from tracking import maskutil

        x, y, w, h = maskutil.to_bbox(rle)
        if w == 0 and h == 0:
            return None
        return np.array([x, y, x + w, y + h], dtype=np.float32)
    except Exception:
        return None


class Detection:
    __slots__ = (
        "class_id",
        "score",
        "bbox",
        "centroid",
        "area",
        "aspect_ratio",
        "mask_rle",
        "query_index",
        "query_embed",
        "mask_pooled",
        "reid",
        "raw",
    )

    def __init__(
        self,
        class_id: int,
        score: float,
        bbox: np.ndarray,
        centroid: Optional[np.ndarray] = None,
        area: Optional[float] = None,
        aspect_ratio: Optional[float] = None,
        mask_rle: Optional[Dict] = None,
        query_index: Optional[int] = None,
        query_embed: Optional[np.ndarray] = None,
        mask_pooled: Optional[np.ndarray] = None,
        reid: Optional[np.ndarray] = None,
        raw: Optional[Dict] = None,
    ):
        self.class_id = int(class_id)
        self.score = float(score)
        self.bbox = np.asarray(bbox, dtype=np.float32)
        if centroid is None:
            centroid = np.array(
                [(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2],
                dtype=np.float32,
            )
        self.centroid = np.asarray(centroid, dtype=np.float32)
        w = max(self.bbox[2] - self.bbox[0], 1e-3)
        h = max(self.bbox[3] - self.bbox[1], 1e-3)
        self.area = float(area) if area is not None else float(w * h)
        self.aspect_ratio = float(aspect_ratio) if aspect_ratio is not None else float(w / h)
        self.mask_rle = mask_rle
        self.query_index = query_index
        self.query_embed = _as_vec(query_embed)
        self.mask_pooled = _as_vec(mask_pooled)
        self.reid = _as_vec(reid)
        self.raw = raw

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def scale(self) -> float:
        return float(np.sqrt(max(self.area, 1.0)))

    def appearance(self, source: str) -> Optional[np.ndarray]:
        return getattr(self, source, None)

    @classmethod
    def from_dict(cls, d: Dict) -> "Detection":
        bbox = d.get("bbox")
        if bbox is None and d.get("mask_rle") is not None:
            bbox = bbox_from_mask_rle(d["mask_rle"])
        if bbox is None:
            raise ValueError("Detection has neither bbox nor decodable mask_rle")
        return cls(
            class_id=d["class_id"],
            score=d.get("score", 1.0),
            bbox=bbox,
            centroid=d.get("centroid"),
            area=d.get("area"),
            aspect_ratio=d.get("aspect_ratio"),
            mask_rle=d.get("mask_rle"),
            query_index=d.get("query_index"),
            query_embed=d.get("query_embed"),
            mask_pooled=d.get("mask_pooled"),
            reid=d.get("reid"),
            raw=d,
        )


def _as_vec(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _l2(v: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(v, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 1e-6 else arr


class Track:
    _count = 0

    def __init__(self, detection: Detection, frame_id: int, min_hits: int = 3):
        Track._count += 1
        self.track_id = Track._count

        self.class_id = detection.class_id
        self.state = TrackState.TENTATIVE
        self.min_hits = min_hits

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.frames_since_update = 0

        self.start_frame = frame_id
        self.last_frame_id = frame_id
        self.last_detection = detection

        self.bbox = detection.bbox.copy()
        self.centroid = detection.centroid.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self._prev_centroid = detection.centroid.copy()
        self._prev_frame_id = frame_id

        self.kf = None
        self.appearance_protos: Dict[str, np.ndarray] = {}
        self._update_appearance(detection, ema=0.0)

        if self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED

    def predict_centroid(self, frame_id: int, delta_t_cap: int) -> np.ndarray:
        steps = self._steps_to(frame_id)
        steps = min(steps, delta_t_cap)
        return self.centroid + self.velocity * steps

    def _steps_to(self, frame_id: int) -> float:
        return max(1.0, float(self.time_since_update + 1))

    def update(self, detection: Detection, frame_id: int, appearance_ema: float = 0.9):
        dt = max(1.0, float(frame_id - self.last_frame_id))
        self.velocity = (detection.centroid - self.centroid) / dt
        self._prev_centroid = self.centroid.copy()
        self._prev_frame_id = self.last_frame_id

        self.bbox = detection.bbox.copy()
        self.centroid = detection.centroid.copy()
        self.last_detection = detection
        self.last_frame_id = frame_id

        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.frames_since_update = 0

        if self.state == TrackState.TENTATIVE and self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

        self._update_appearance(detection, appearance_ema)

    def _update_appearance(self, detection: Detection, ema: float):
        for src in APPEARANCE_SOURCES:
            feat = getattr(detection, src, None)
            if feat is None:
                continue
            prev = self.appearance_protos.get(src)
            if prev is None:
                self.appearance_protos[src] = _l2(feat)
            else:
                self.appearance_protos[src] = _l2(ema * prev + (1.0 - ema) * feat)

    def mark_missed(self, frame_id: int):
        self.age += 1
        self.time_since_update += 1
        self.frames_since_update = frame_id - self.last_frame_id
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.REMOVED
        elif self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST

    def is_active(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)

    def is_output(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @staticmethod
    def reset_id_counter():
        Track._count = 0
