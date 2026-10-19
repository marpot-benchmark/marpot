
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np

from tracking.base_tracker import BaseTracker
from tracking.byte_tracker import ByteTracker
from tracking.config import TrackerConfig
from tracking.motion_iou_tracker import MotionIoUTracker
from tracking.sea_prior import SeaPrior
from tracking.structures import Detection


def build_tracker(config: Union[TrackerConfig, Dict, None]) -> BaseTracker:
    cfg = config if isinstance(config, TrackerConfig) else TrackerConfig.from_dict(config or {})
    if cfg.type == "motion_iou":
        return MotionIoUTracker(cfg)
    if cfg.type == "byte":
        return ByteTracker(cfg)
    raise ValueError(f"Unknown tracker type: {cfg.type!r} (expected motion_iou | byte)")


def track_sequence(
    frames: List[Dict],
    config: Union[TrackerConfig, Dict, None],
    frame_loader: Optional[Callable[[Dict], np.ndarray]] = None,
) -> List[Dict]:
    cfg = config if isinstance(config, TrackerConfig) else TrackerConfig.from_dict(config or {})
    tracker = build_tracker(cfg)
    sea_prior = SeaPrior(cfg)

    gmc = None
    if cfg.gmc_enabled:
        from tracking.gmc import GMC

        gmc = GMC(method=cfg.gmc_method)
        if not gmc.available:
            import warnings

            warnings.warn(
                "gmc_enabled=True but OpenCV (cv2) is unavailable; global motion "
                "compensation is a no-op. Install opencv-python to enable it.",
                RuntimeWarning,
            )
        elif frame_loader is None:
            import warnings

            warnings.warn(
                "gmc_enabled=True but no frame_loader was provided; GMC needs the "
                "raw frames. Pass --images_root to run_track.py.",
                RuntimeWarning,
            )

    out_frames: List[Dict] = []
    for fr in frames:
        frame_id = int(fr["frame_id"])
        det_dicts = fr.get("detections", [])
        dets = [Detection.from_dict(d) for d in det_dicts]

        dets = sea_prior.apply(dets, fr.get("seasurface_rle"))

        if gmc is not None and frame_loader is not None:
            img = frame_loader(fr)
            affine = gmc.apply(img)
            tracker.apply_gmc(affine)

        active = tracker.update(dets, frame_id)

        out = dict(fr)
        out_dets = []
        for t in active:
            d = t.last_detection
            src = d.raw if d.raw is not None else {}
            out_dets.append(
                {
                    **{k: v for k, v in src.items() if k not in ("track_id",)},
                    "track_id": t.track_id,
                    "class_id": t.class_id,
                    "score": d.score,
                    "bbox": [float(x) for x in d.bbox],
                }
            )
        out["detections"] = out_dets
        out_frames.append(out)

    return out_frames
