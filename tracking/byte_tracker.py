
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from tracking import association as assoc
from tracking.base_tracker import BaseTracker
from tracking.config import TrackerConfig
from tracking.kalman import KalmanBoxTracker
from tracking.structures import Detection, Track, TrackState


class ByteTracker(BaseTracker):
    def __init__(self, config: TrackerConfig):
        super().__init__(config)

    def create_track(self, detection: Detection, frame_id: int) -> Track:
        t = super().create_track(detection, frame_id)
        if self.cfg.kalman_enabled:
            t.kf = KalmanBoxTracker(detection.bbox, nsa=self.cfg.nsa_kalman)
        return t

    def _apply_match(self, track: Track, det: Detection, frame_id: int):
        steps = min(track.time_since_update + 1, self.cfg.delta_t_cap)
        recovering = track.time_since_update > 0
        prev_centroid = track.centroid.copy()
        if recovering and self.cfg.oru_enabled and track.kf is not None:
            self._oru_reupdate(track, det, frame_id)
            steps = 1
        track.update(det, frame_id, self.cfg.appearance_ema)
        if track.kf is not None:
            track.kf.update(det.bbox, det.score, steps)
            if self.cfg.ocm_enabled:
                track.kf.mean[4:6] = (det.centroid - prev_centroid) / max(steps, 1)

    def _oru_reupdate(self, track: Track, det: Detection, frame_id: int):
        gap = min(frame_id - track.last_frame_id, self.cfg.delta_t_cap)
        if gap <= 1:
            return
        last = np.asarray(track.last_detection.bbox, dtype=np.float32)
        cur = np.asarray(det.bbox, dtype=np.float32)
        score = track.last_detection.score
        for s in range(1, gap):
            alpha = s / gap
            virt = last * (1.0 - alpha) + cur * alpha
            track.kf.update(virt, score, 1.0)

    def update(self, detections: List[Detection], frame_id: int) -> List[Track]:
        self.frame_count += 1
        self.handle_scene_cut(frame_id)

        if not self.cfg.byte_enabled:
            dets = self.filter_by_score(detections)
            self._single_stage(dets, frame_id)
            self.prune()
            return self.outputs()

        kept = [d for d in detections if d.score >= self.cfg.low_thr]
        high = [d for d in kept if d.score >= self.cfg.high_thr]
        low = [d for d in kept if d.score < self.cfg.high_thr]

        active = [t for t in self.tracks if t.is_active()]

        cost = self.compute_cost(high, active, frame_id)
        m1, u_high, u_trk1 = assoc.linear_assignment(cost, self.cfg.match_cost_threshold)
        for di, ti in m1:
            self._apply_match(active[ti], high[di], frame_id)

        remaining_tracks = [active[ti] for ti in u_trk1]

        stage2_tracks = [t for t in remaining_tracks if t.state != TrackState.TENTATIVE]
        cost2 = self.compute_cost(low, stage2_tracks, frame_id)
        m2, _, u_trk2 = assoc.linear_assignment(cost2, self.cfg.match_cost_threshold)
        for di, ti in m2:
            self._apply_match(stage2_tracks[ti], low[di], frame_id)

        still_unmatched = [stage2_tracks[ti] for ti in u_trk2]

        leftover_high = [high[di] for di in u_high]
        if self.cfg.ocr_enabled and leftover_high and still_unmatched:
            recovered_dets = self._ocr_stage(leftover_high, still_unmatched, frame_id)
            leftover_high = [d for k, d in enumerate(leftover_high) if k not in recovered_dets]

        for d in leftover_high:
            self.tracks.append(self.create_track(d, frame_id))

        for t in active:
            if t.last_frame_id != frame_id:
                t.mark_missed(frame_id)

        self.prune()
        return self.outputs()

    def _single_stage(self, dets: Sequence[Detection], frame_id: int):
        active = [t for t in self.tracks if t.is_active()]
        cost = self.compute_cost(dets, active, frame_id)
        matches, u_dets, u_trks = assoc.linear_assignment(
            cost, self.cfg.match_cost_threshold
        )
        for di, ti in matches:
            self._apply_match(active[ti], dets[di], frame_id)
        for di in u_dets:
            self.tracks.append(self.create_track(dets[di], frame_id))
        for ti in u_trks:
            active[ti].mark_missed(frame_id)

    def _ocr_stage(self, dets, tracks, frame_id) -> set:
        last_bboxes = [t.last_detection.bbox for t in tracks]
        iou = assoc.bbox_iou_matrix(dets, last_bboxes)
        scales = [max(np.sqrt(max(t.last_detection.area, 1.0)), 1.0) for t in tracks]
        last_centroids = [t.last_detection.centroid for t in tracks]
        cdist = assoc.center_dist_matrix(dets, last_centroids, scales)
        cost = (1.0 - iou) + np.minimum(cdist, 1.0)
        for i, d in enumerate(dets):
            for j, t in enumerate(tracks):
                if (not self.cfg.class_agnostic) and d.class_id != t.class_id:
                    cost[i, j] = 1e6
                elif cdist[i, j] > self.cfg.center_dist_factor:
                    cost[i, j] = 1e6
        matches, _, _ = assoc.linear_assignment(cost, self.cfg.match_cost_threshold)
        recovered = set()
        for di, ti in matches:
            self._apply_match(tracks[ti], dets[di], frame_id)
            recovered.add(di)
        return recovered
