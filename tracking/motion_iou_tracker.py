
from __future__ import annotations

from typing import List

from tracking.association import linear_assignment
from tracking.base_tracker import BaseTracker
from tracking.config import TrackerConfig
from tracking.structures import Detection, Track


class MotionIoUTracker(BaseTracker):
    def __init__(self, config: TrackerConfig):
        super().__init__(config)

    def update(self, detections: List[Detection], frame_id: int) -> List[Track]:
        self.frame_count += 1
        self.handle_scene_cut(frame_id)

        dets = self.filter_by_score(detections)
        active = [t for t in self.tracks if t.is_active()]

        cost = self.compute_cost(dets, active, frame_id)
        matches, unmatched_dets, unmatched_trks = linear_assignment(
            cost, max_cost=self.cfg.match_cost_threshold
        )

        for di, ti in matches:
            active[ti].update(dets[di], frame_id, self.cfg.appearance_ema)

        for di in unmatched_dets:
            self.tracks.append(self.create_track(dets[di], frame_id))

        for ti in unmatched_trks:
            active[ti].mark_missed(frame_id)

        self.prune()
        return self.outputs()
