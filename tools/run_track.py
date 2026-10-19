#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tracking.build import track_sequence
from tracking.config import TrackerConfig
from tracking.io import (
    group_by_sequence,
    read_detections_jsonl,
    write_mot_txt,
    write_mots_txt,
    write_tracks_jsonl,
)


def load_tracker_config(path: Optional[str]) -> TrackerConfig:
    if path is None:
        return TrackerConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if "tracker" in data:
        data = data["tracker"]
    return TrackerConfig.from_dict(data)


def make_frame_loader(images_root: Optional[str]):
    if images_root is None:
        return None
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    root = Path(images_root)

    def loader(frame):
        return np.array(Image.open(root / frame["file"]).convert("RGB"))

    return loader


def main():
    p = argparse.ArgumentParser(description="Run a tracker over exported detections")
    p.add_argument("--dets", required=True, help="detection JSONL")
    p.add_argument("--tracker_config", default=None, help="configs/tracking/*.yaml")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--images_root", default=None, help="needed only when gmc_enabled")
    args = p.parse_args()

    cfg = load_tracker_config(args.tracker_config)
    print(f"[run_track] tracker config: {cfg.type} | byte={cfg.byte_enabled} "
          f"kalman={cfg.kalman_enabled} query={cfg.use_query_embed} "
          f"appearance={cfg.appearance_enabled} gmc={cfg.gmc_enabled} "
          f"sea_prior={cfg.sea_prior_enabled}")

    frames = read_detections_jsonl(args.dets)
    sequences = group_by_sequence(frames)
    frame_loader = make_frame_loader(args.images_root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tracked = []
    for seq_id, seq_frames in sequences.items():
        tracked = track_sequence(seq_frames, cfg, frame_loader=frame_loader)

        if cfg.min_track_len > 1:
            tracked = _drop_short_tracks(tracked, cfg.min_track_len)

        all_tracked.extend(tracked)

        write_mot_txt(str(out_dir / "mot" / f"{seq_id}.txt"), tracked)
        write_mots_txt(str(out_dir / "mots" / f"{seq_id}.txt"), tracked)
        n_ids = len({d["track_id"] for fr in tracked for d in fr["detections"]})
        print(f"[run_track] seq {seq_id}: {len(tracked)} frames, {n_ids} track ids")

    write_tracks_jsonl(str(out_dir / "tracks.jsonl"), all_tracked)
    print(f"[run_track] wrote {out_dir}/tracks.jsonl ({len(all_tracked)} frames)")


def _drop_short_tracks(frames, min_len):
    counts = {}
    for fr in frames:
        for d in fr["detections"]:
            counts[d["track_id"]] = counts.get(d["track_id"], 0) + 1
    for fr in frames:
        fr["detections"] = [
            d for d in fr["detections"] if counts.get(d["track_id"], 0) >= min_len
        ]
    return frames


if __name__ == "__main__":
    main()
