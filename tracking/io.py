
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_seq_frame(file_name: str) -> Tuple[str, int]:
    path = Path(file_name)
    seq_id = path.parent.name
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse frame number from {file_name}")
    return seq_id, int(match.group(1))


def read_detections_jsonl(path: str) -> List[Dict]:
    frames: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    frames.sort(key=lambda fr: int(fr.get("frame_id", _frame_id_of(fr))))
    return frames


def _frame_id_of(frame: Dict) -> int:
    if "frame_id" in frame:
        return int(frame["frame_id"])
    return parse_seq_frame(frame["file"])[1]


def group_by_sequence(frames: Iterable[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for fr in frames:
        seq = str(fr.get("seq")) if fr.get("seq") is not None else parse_seq_frame(fr["file"])[0]
        groups.setdefault(seq, []).append(fr)
    for seq in groups:
        groups[seq].sort(key=_frame_id_of)
    return groups


def write_tracks_jsonl(path: str, frames: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")


def write_mot_txt(path: str, frames: List[Dict], class_id: Optional[int] = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for idx, fr in enumerate(frames, start=1):
            for det in fr.get("detections", []):
                if det.get("track_id") is None:
                    continue
                if class_id is not None and det.get("class_id") != class_id:
                    continue
                x1, y1, x2, y2 = det["bbox"]
                f.write(
                    f"{idx},{det['track_id']},{x1:.2f},{y1:.2f},"
                    f"{x2 - x1:.2f},{y2 - y1:.2f},{det.get('score', 1.0):.4f},-1,-1,-1\n"
                )


def write_mots_txt(path: str, frames: List[Dict], class_id_to_mots: Optional[Dict] = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for idx, fr in enumerate(frames, start=1):
            for det in fr.get("detections", []):
                if det.get("track_id") is None or det.get("mask_rle") is None:
                    continue
                rle = det["mask_rle"]
                h, w = rle["size"]
                counts = rle["counts"]
                if isinstance(counts, bytes):
                    counts = counts.decode("ascii")
                cls = det.get("class_id", 0) + 1
                obj_id = cls * 1000 + int(det["track_id"])
                f.write(f"{idx} {obj_id} {cls} {h} {w} {counts}\n")
