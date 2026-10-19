
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from tracking import association as assoc
from tracking.config import TrackerConfig
from tracking.structures import Detection, Track, TrackState


class BaseTracker:
    def __init__(self, config: TrackerConfig):
        self.cfg = config
        self.tracks: List[Track] = []
        self.frame_count = 0
        Track.reset_id_counter()

    def update(self, detections: List[Detection], frame_id: int) -> List[Track]:
        raise NotImplementedError

    def filter_by_score(self, detections: Sequence[Detection]) -> List[Detection]:
        return [
            d for d in detections if d.score >= self.cfg.min_score_for_class(d.class_id)
        ]

    def handle_scene_cut(self, frame_id: int):
        for t in self.tracks:
            gap = frame_id - t.last_frame_id
            if gap > self.cfg.scene_cut_gap:
                t.velocity[:] = 0.0
                if t.state == TrackState.LOST:
                    t.state = TrackState.REMOVED
                if t.kf is not None and hasattr(t.kf, "reset_velocity"):
                    t.kf.reset_velocity()

    def prune(self):
        kept = []
        for t in self.tracks:
            if t.state == TrackState.REMOVED:
                continue
            if (
                t.time_since_update > self.cfg.max_age_samples
                or t.frames_since_update > self.cfg.max_age_frames
            ):
                continue
            kept.append(t)
        self.tracks = kept

    def outputs(self) -> List[Track]:
        return [
            t
            for t in self.tracks
            if t.is_output()
            and t.time_since_update == 0
            and t.hits >= self.cfg.min_track_len
        ]

    def predicted_geometry(self, tracks: Sequence[Track], frame_id: int):
        centroids, bboxes, scales = [], [], []
        cap = self.cfg.delta_t_cap
        for t in tracks:
            if self.cfg.motion_free_ablation:
                c = t.centroid.copy()
                b = t.bbox.copy()
            elif t.kf is not None:
                steps = min(max(1, t.time_since_update + 1), cap)
                b = t.kf.predict_bbox(steps)
                c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], dtype=np.float32)
            else:
                c = t.predict_centroid(frame_id, cap)
                shift = c - t.centroid
                b = t.bbox + np.array([shift[0], shift[1], shift[0], shift[1]])
            centroids.append(c)
            bboxes.append(b)
            scales.append(max(np.sqrt(max(t.last_detection.area, 1.0)), 1.0))
        return centroids, bboxes, scales

    def compute_cost(
        self, dets: Sequence[Detection], tracks: Sequence[Track], frame_id: int
    ) -> np.ndarray:
        n, m = len(dets), len(tracks)
        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float32)

        centroids, bboxes, scales = self.predicted_geometry(tracks, frame_id)
        w = self.cfg.cost_weights

        iou = None
        if self.cfg.iou_kind == "mask":
            iou = assoc.mask_iou_matrix(dets, tracks)
        if iou is None:
            iou = assoc.bbox_iou_matrix(dets, bboxes)
        iou_thr = (
            self.cfg.mask_iou_threshold
            if self.cfg.iou_kind == "mask"
            else self.cfg.bbox_iou_threshold
        )

        cdist = assoc.center_dist_matrix(dets, centroids, scales)
        achange = assoc.area_change_matrix(dets, tracks)

        cost = w.iou * (1.0 - iou) + w.center * np.minimum(cdist, 1.0)

        if w.area > 0:
            cost = cost + w.area * np.minimum((achange - 1.0) / 3.0, 1.0)

        if self.cfg.use_query_embed and w.query > 0:
            qc = assoc.query_cosine_matrix(dets, tracks)
            if qc is not None:
                cost = cost + w.query * qc

        if self.cfg.appearance_enabled and w.appearance > 0:
            ac = assoc.cosine_matrix(dets, tracks, self.cfg.appearance_source)
            if ac is not None:
                cost = cost + w.appearance * ac

        for i, d in enumerate(dets):
            for j, t in enumerate(tracks):
                forbidden = False
                if (not self.cfg.class_agnostic) and d.class_id != t.class_id:
                    forbidden = True
                elif cdist[i, j] > self.cfg.center_dist_factor:
                    forbidden = True
                elif achange[i, j] > self.cfg.area_change_max:
                    forbidden = True
                elif iou[i, j] < iou_thr and cdist[i, j] > self.cfg.center_dist_factor * 0.5:
                    forbidden = True
                if forbidden:
                    cost[i, j] = 1e6
        return cost

    def apply_gmc(self, affine: Optional[np.ndarray]):
        if affine is None:
            return
        from tracking.gmc import warp_bbox, warp_points

        for t in self.tracks:
            if not t.is_active():
                continue
            t.bbox = warp_bbox(t.bbox, affine)
            t.centroid = warp_points(t.centroid, affine)
            t._prev_centroid = warp_points(t._prev_centroid, affine)
            t.velocity = (affine[:, :2] @ t.velocity.reshape(2, 1)).reshape(2)
            if t.kf is not None:
                c = warp_points(t.kf.mean[:2], affine)
                t.kf.mean[0], t.kf.mean[1] = float(c[0]), float(c[1])

    def create_track(self, detection: Detection, frame_id: int) -> Track:
        return Track(detection, frame_id, min_hits=self.cfg.min_hits)
