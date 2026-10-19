#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _bbox_iou(a, b) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = w * h
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _iou_matrix(gts, preds, iou_kind):
    if not gts or not preds:
        return np.zeros((len(gts), len(preds)))
    if iou_kind == "mask" and all(g.get("mask_rle") for g in gts) and all(
        p.get("mask_rle") for p in preds
    ):
        from tracking import maskutil

        return maskutil.iou_matrix([g["mask_rle"] for g in gts], [p["mask_rle"] for p in preds])
    m = np.zeros((len(gts), len(preds)))
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            m[i, j] = _bbox_iou(g["bbox"], p["bbox"])
    return m


def _seq_of(fr: dict) -> str:
    if fr.get("seq") is not None:
        return str(fr["seq"])
    f = fr.get("file")
    if f:
        from tracking.io import parse_seq_frame

        return parse_seq_frame(f)[0]
    return ""


def _load_indexed(path: str) -> Dict[tuple, List[dict]]:
    by_frame: Dict[tuple, List[dict]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fr = json.loads(line)
            seq = _seq_of(fr)
            dets = fr.get("detections", [])
            for d in dets:
                if d.get("track_id") is not None:
                    d["track_id"] = f"{seq}:{d['track_id']}"
            by_frame.setdefault((seq, int(fr["frame_id"])), []).extend(dets)
    return by_frame


def evaluate(gt_path, pred_path, iou_thr=0.5, iou_kind="bbox", class_id=None) -> Dict:
    gt_by_frame = _load_indexed(gt_path)
    pred_by_frame = _load_indexed(pred_path)
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    tp = fp = fn = idsw = 0
    iou_sum = 0.0
    n_gt = n_pred = 0
    last_match: Dict[object, object] = {}
    cooc: Dict[tuple, int] = {}

    for fr in frames:
        gts = [d for d in gt_by_frame.get(fr, []) if class_id is None or d.get("class_id") == class_id]
        preds = [d for d in pred_by_frame.get(fr, []) if class_id is None or d.get("class_id") == class_id]
        n_gt += len(gts)
        n_pred += len(preds)

        iou = _iou_matrix(gts, preds, iou_kind)
        matches = []
        if len(gts) and len(preds):
            cost = 1.0 - iou
            if linear_sum_assignment is None:
                raise ImportError("scipy required for evaluation")
            ri, ci = linear_sum_assignment(cost)
            matches = [(r, c) for r, c in zip(ri, ci) if iou[r, c] >= iou_thr]

        matched_g, matched_p = set(), set()
        cur_match: Dict[int, int] = {}
        for r, c in matches:
            gid = gts[r]["track_id"]
            pid = preds[c]["track_id"]
            tp += 1
            iou_sum += iou[r, c]
            matched_g.add(r)
            matched_p.add(c)
            cur_match[gid] = pid
            cooc[(gid, pid)] = cooc.get((gid, pid), 0) + 1
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
        fn += len(gts) - len(matched_g)
        fp += len(preds) - len(matched_p)
        last_match.update(cur_match)

    idtp = _idf1_idtp(cooc)
    idfp = n_pred - idtp
    idfn = n_gt - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1e-9)

    mota = 1.0 - (fn + fp + idsw) / max(n_gt, 1)
    motp = iou_sum / max(tp, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    if n_gt == 0:
        return {
            "n_gt": 0, "n_pred": n_pred, "TP": tp, "FP": fp, "FN": fn,
            "IDsw": idsw, "MOTA": "-", "MOTP": "-",
            "Recall": "-", "Precision": "-", "IDF1": "-", "IDTP": idtp,
        }

    return {
        "n_gt": n_gt, "n_pred": n_pred, "TP": tp, "FP": fp, "FN": fn,
        "IDsw": idsw, "MOTA": round(mota, 4), "MOTP": round(motp, 4),
        "Recall": round(recall, 4), "Precision": round(precision, 4),
        "IDF1": round(idf1, 4), "IDTP": idtp,
    }


def _idf1_idtp(cooc: Dict[tuple, int]) -> int:
    if not cooc:
        return 0
    gt_ids = sorted({g for g, _ in cooc})
    pred_ids = sorted({p for _, p in cooc})
    gi = {g: i for i, g in enumerate(gt_ids)}
    pi = {p: i for i, p in enumerate(pred_ids)}
    profit = np.zeros((len(gt_ids), len(pred_ids)))
    for (g, p), c in cooc.items():
        profit[gi[g], pi[p]] = c
    if linear_sum_assignment is None:
        return int(profit.max(axis=1).sum())
    ri, ci = linear_sum_assignment(-profit)
    return int(profit[ri, ci].sum())


def main():
    p = argparse.ArgumentParser(description="Evaluate tracks (CLEAR-MOT + IDF1)")
    p.add_argument("--gt", required=True, help="GT JSONL with track_id")
    p.add_argument("--pred", required=True, help="tracker output JSONL (tracks.jsonl)")
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--iou_kind", default="bbox", choices=["bbox", "mask"])
    p.add_argument("--classes", type=int, nargs="*", default=[0, 1])
    p.add_argument("--class_names", nargs="*", default=["ship", "obstacle"])
    p.add_argument("--out", default=None, help="optional path to write a JSON summary")
    args = p.parse_args()

    results = {"all": evaluate(args.gt, args.pred, args.iou_thr, args.iou_kind, None)}
    for cid, name in zip(args.classes, args.class_names):
        results[name] = evaluate(args.gt, args.pred, args.iou_thr, args.iou_kind, cid)

    print(f"\n=== Track eval ({args.iou_kind}-IoU >= {args.iou_thr}) ===")
    hdr = ["MOTA", "IDF1", "Recall", "Precision", "IDsw", "FP", "FN"]
    print(f"{'split':>10} | " + " | ".join(f"{h:>9}" for h in hdr))
    for name, r in results.items():
        print(f"{name:>10} | " + " | ".join(f"{r[h]:>9}" for h in hdr))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[eval_track] wrote {args.out}")


if __name__ == "__main__":
    main()
