# MarPOT

Code for **MarPOT: A High-Resolution Maritime Panoptic Benchmark and a Training-Free Re-Observation Baseline for Small Instances**.

Project page (dataset after registration, checkpoints, reference results): https://marpot-benchmark.github.io/

The repository contains

- the MarPOT evaluation protocol: PQ, PQ<sub>Th</sub>, PQ<sub>St</sub> and small/medium/large-instance recall in native pixels, paired clip-bootstrap intervals, and a standalone evaluator for any panoptic prediction;
- **re-observation**: the full-frame pass, query-evidence window selection, local passes on 1120-pixel windows and stuff-anchored composition, for EoMT and Mask2Former;
- the seven association configurations and their evaluation (HOTA, IDF1, MOTA, identity switches);
- training of the EoMT predictor (DINOv2-L) on MarPOT.

The code builds on [EoMT](https://github.com/tue-mps/eomt).

## Layout

```text
models/ datasets/ training/        EoMT (DINOv2-L), the MarPOT dataset loader, losses and metrics
tile_geometry.py                   candidate windows (1120 px, overlap 0.2)
tools/eomt_infer.py                full-frame pass, query evidence, local raw-query detections
tools/reobs_core.py                greedy window selection with evidence clearing
tools/tiled_panoptic_infer.py      local passes, IoMin union, stuff-anchored composition
tools/analyze_panoptic_errors.py   inference and native-resolution scoring with per-instance records
tools/selector_adapter.py          predictive-entropy window selector
tools/m2f_backend.py, m2f/         Mask2Former (Swin-S) backend for re-observation
evaluation/specs.py                every evaluated configuration as a spec
evaluation/run_eval.py             one spec on one split (or a shard of it)
evaluation/postprocess.py          pooling of shards; parameter selection on validation
evaluation/paper_statistics.py     paired clip bootstrap, split statistics, size bins, residual misses
evaluation/prepare_bins.py, binned_ci.py   paired intervals per area bin and per class and size
evaluation/native_pq.py            standalone evaluator for panoptic predictions (PNG + JSON)
evaluation/latency.py              runtime protocol (batch 1, 128 test frames x 3 rounds)
tracking/, tools/run_track.py, tools/eval_track.py, tools/hota.py   association configurations and metrics
tracking_eval/                     association evaluation on the test split
train.py, training/runtime.py      EoMT training (one GPU, two virtual ranks, bf16)
configs/eomt/                      EoMT configurations (configs/eomt/eomt_large_896_joint_scratch_cp.yaml: predictor of the paper)
configs/tracking/                  the seven association configurations
configs/m2f/                       Mask2Former (Swin-S) configuration
```

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements_track.txt   # association evaluation
pip install -r requirements_m2f.txt     # Mask2Former only (MMDetection 3.3.0, mmcv 2.1.0 with CUDA ops)
```

Install the PyTorch build that matches your CUDA version. The results of the paper were produced with Python 3.12, PyTorch 2.8 (CUDA 12.9), Lightning 2.5.1, torchmetrics 1.8.2, timm 1.0.15 and MMDetection 3.3.0. torchmetrics must be 1.8.2 (the native panoptic metric uses its internal functions), and the association evaluation needs NumPy < 2. Python 3.12 or later reproduces the pooled PQ bit for bit.

## Data

Download MarPOT v6.0.1 from the project page and arrange it as

```text
data_panoptic_track/
├── annotations/panoptic_{train,val,test}.json
├── images/{train,val,test}/<video_id>/<frame>.jpg
└── panoptic/{train,val,test}/<video_id>/<frame>.png
```

The annotations follow COCO panoptic. `images[]` carry `video_id`, `frame_id` and `original_video_id`; `segments_info` carry `iscrowd` and `track_id` (-1: no identity). Categories: 1 ship, 2 obstacle (things), 3 seasurface, 4 backdrop (stuff). Panoptic PNGs encode the segment id as R + 256·G + 256²·B. The split has 255 / 13 / 56 clips with 3,984 / 238 / 2,675 frames; the test clips form 91 continuous sequences. Bootstrap intervals resample the 56 test clips (`original_video_id`), drawing all sequences of a clip together.

## Checkpoints

The checkpoints of the paper are in the download folder of the project page:

| Checkpoint | Used for | File | SHA-256 |
| --- | --- | --- | --- |
| EoMT (DINOv2-L), 896-pixel input | all EoMT results | `inference_state.pt` | `104a4e4930c92873cda5c16072b184a3013709d176a75d9c32519904083f004b` |
| Mask2Former (Swin-S) | Mask2Former rows | `best_coco_panoptic_PQ_epoch_28.pth` | `785e0b1eea4cd7042d494d681222f0755cfab212ec7b4bb9a92e43c118e3e811` |

Training starts from the COCO-panoptic EoMT-L 640 weights (DINOv2) of the [EoMT model zoo](https://github.com/tue-mps/eomt), expected at `weights/coco_panoptic_eomt_large_640/pytorch_model.bin` unless `--init-checkpoint` is given.

## Quick start

```bash
python evaluation/specs.py --out configs/specs

python evaluation/run_eval.py --data-root DATA --config configs/eomt/eomt_large_896_joint_scratch_cp.yaml \
  --checkpoint inference_state.pt --spec configs/specs/eomt_test_query.json --out runs/eomt_test_query --export
python evaluation/run_eval.py --data-root DATA --config configs/eomt/eomt_large_896_joint_scratch_cp.yaml \
  --checkpoint inference_state.pt --spec configs/specs/eomt_test_full896.json --out runs/eomt_test_full896

python evaluation/postprocess.py aggregate --spec configs/specs/eomt_test_query.json \
  --inputs runs/eomt_test_query/summary.json --out runs/aggregate_eomt_test_query
python evaluation/postprocess.py aggregate --spec configs/specs/eomt_test_full896.json \
  --inputs runs/eomt_test_full896/summary.json --out runs/aggregate_eomt_test_full896

python evaluation/paper_statistics.py paired --data DATA --out stats/query_vs_full896 \
  --baseline runs/aggregate_eomt_test_full896/summary.json --candidate runs/aggregate_eomt_test_query/summary.json
```

`run_eval.py` accepts `--start/--end` to split a run into shards (frame order of `annotations[]`); pool the shards with `postprocess.py aggregate --inputs <all shard summaries>`.

### Evaluating your own predictions

```bash
python evaluation/native_pq.py --data DATA --split test --pred-json predictions.json --pred-dir predictions/ --out metrics.json
```

Predictions follow COCO panoptic: one PNG per annotated frame at native resolution (R + 256·G + 256²·B, 0 = void) and a JSON whose `annotations[]` give `file_name` (`<video_id>/<frame>.png`, optionally prefixed by the split) or `image_id`, and `segments_info` with `id` and `category_id` (1–4). The evaluator reports PQ, PQ<sub>Th</sub>, PQ<sub>St</sub>, SQ, RQ, R<sub>S</sub>, R<sub>M</sub>, R<sub>L</sub> (native-pixel area below 32², below 96², above) and the thing TP, FP and FN; the JSON written by `--out` can be passed to `paper_statistics.py paired`. Please report the items listed on the project page: the official split, model selection on validation, input preprocessing and resolution, the number of local passes, and PQ<sub>Th</sub> and R<sub>S</sub> together with the number of thing false positives.

## Reproducing the paper

Run `python evaluation/specs.py --out configs/specs` first. Each spec is evaluated with `run_eval.py` and pooled with `postprocess.py aggregate`; paired differences use `paper_statistics.py paired` (10,000 resamples, seed 0).

| Result | Specs and commands |
| --- | --- |
| Split statistics | `paper_statistics.py dataset --data DATA` |
| Test-time computation, EoMT rows | `eomt_test_{full896,full_thr03,full1344,full1792,random0,random1,random2,center,query,all}`; tokens per frame are (K+1)(S/16)² |
| Test-time computation, Mask2Former rows | calibration grid `m2f_val_*` and `postprocess.py select`, then `m2f_test_{full,query,all}` with `--selection` and `--m2f-config configs/m2f/mask2former_swin_s_marpot.py --m2f-checkpoint best_coco_panoptic_PQ_epoch_28.pth` |
| Paired differences and intervals | `paper_statistics.py paired --baseline <reference> --candidate <row>`; the three random seeds are passed together and averaged after scoring |
| Recall by instance size | `paper_statistics.py figures` on the shards of `eomt_test_full896` and `eomt_test_query`; intervals with `prepare_bins.py` and `binned_ci.py` |
| Window placement, predictive entropy, retained evidence | `eomt_test_{horizon,center,entropy,retained}`; selected windows in `summary/frame_routes.jsonl` |
| Window size | `eomt_test_{query,all}_w{560,840,1400}` against `eomt_test_{query,all}` |
| Runtimes and stage breakdown | `latency.py --data-root DATA --config ... --checkpoint inference_state.pt --out runs/latency [--stages]` on an otherwise idle GPU |
| Association | `eomt_test_full896` and `eomt_test_query`, then `tracking_eval/prepare_truth.py`, `tracking_eval/build_cache_manifest.py` (once per run) and `tracking_eval/run.py --tracker all` |
| Residual misses | `eomt_test_query_raw`, then `paper_statistics.py residual --split test --data DATA --inputs <shard summaries>` |
| Validation studies (placement, budget, evidence variants) | `eomt_val_*`; variant selection with `postprocess.py select` |
| EoMT training configurations | `train.py --config configs/eomt/<config>.yaml --data-root DATA --out OUT` |

### Association

```bash
python tracking_eval/prepare_truth.py --data-root DATA --out runs/track/truth
python tracking_eval/build_cache_manifest.py --summaries runs/eomt_test_full896/summary.json \
  --checkpoint inference_state.pt --data-root DATA --out runs/track/full_cache.json
python tracking_eval/build_cache_manifest.py --summaries runs/eomt_test_query/summary.json \
  --checkpoint inference_state.pt --data-root DATA --out runs/track/query_cache.json
python tracking_eval/run.py --truth-manifest runs/track/truth/truth_manifest.json \
  --full-cache-manifest runs/track/full_cache.json --query-cache-manifest runs/track/query_cache.json \
  --tracker all --out runs/track/results
```

Evaluation and association must use the same `--data-root`. The tracker implementations follow ByteTrack, OC-SORT and BoT-SORT; HOTA is computed with TrackEval.

### Training

```bash
python train.py --config configs/eomt/eomt_large_896_joint_scratch_cp.yaml --data-root DATA --out runs/train_896 \
  --init-checkpoint weights/coco_panoptic_eomt_large_640/pytorch_model.bin
```

Training runs on one GPU in bf16; configurations written for two devices are emulated with two virtual ranks. It selects the checkpoint with the best validation PQ; the attention-mask annealing steps of the configuration are recomputed from the actual number of steps. The warm-start configuration `eomt_large_640_joint.yaml` needs `--init-checkpoint` pointing to a panoptic-only 640 run (`eomt_large_640_track.yaml`). The output `inference_state.pt` is the checkpoint used by `run_eval.py`.

## Not included

Baseline training code and third-party code bases (MMSegmentation, Detectron2, PanSR, PEM, FA-PEM, DAPE, SegMAN, GCNet), the PanSR and LaRS adaptations, the near-duplicate analysis, figure scripts and job orchestration.

## License

MIT (see `LICENSE` and `NOTICE`). Files taken from EoMT keep their copyright headers; portions adapted from Apache-2.0 projects are listed in `NOTICE`. The dataset is released under the MarPOT Terms of Use on the project page.

## Citation

```bibtex
@misc{marpot2026,
  title        = {{MarPOT}: A High-Resolution Maritime Panoptic Benchmark and a Training-Free Re-Observation Baseline for Small Instances},
  author       = {Qing, Yuhao and Wang, Yueying and Zhang, Weidong and Wen, Jie and Xu, Xin},
  year         = {2026},
  howpublished = {\url{https://marpot-benchmark.github.io/}},
  note         = {Dataset version 6.0.1}
}
```
