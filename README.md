# MarPOT

Code for **MarPOT: A High-Resolution Maritime Panoptic Benchmark and a Training-Free Re-Observation Baseline for Small Instances**.

Project page (dataset after registration, checkpoints, reference results): https://marpot-benchmark.github.io/

The repository contains

- the MarPOT evaluation protocol: PQ, PQ<sub>Th</sub>, PQ<sub>St</sub> and small/medium/large-instance recall in native pixels, paired clip-bootstrap intervals, and a standalone evaluator for any panoptic prediction;
- **re-observation**: the full-frame pass, window selection (query evidence, fixed horizon row, image center, random, predictive entropy, detection density, exhaustive tiling), local passes and stuff-anchored composition, for EoMT and Mask2Former;
- the seven association configurations and their evaluation (HOTA, IDF1, MOTA, identity switches);
- training of the EoMT predictor (DINOv2-L) on MarPOT.

The code builds on [EoMT](https://github.com/tue-mps/eomt).

## Layout

```text
models/ datasets/ training/        EoMT (DINOv2-L), the MarPOT dataset loader, losses and metrics
tile_geometry.py                   candidate windows (1120 px, overlap 0.2)
tools/eomt_infer.py                full-frame pass, query evidence, predictive entropy, local raw-query detections
tools/reobs_core.py                greedy window selection with evidence clearing
tools/tiled_panoptic_infer.py      local passes, detection density, IoMin union, stuff-anchored composition
tools/analyze_panoptic_errors.py   inference and native-resolution scoring with per-instance records
tools/m2f_backend.py, m2f/         Mask2Former (Swin-S) backend for re-observation
evaluation/specs.py                every evaluated configuration as a spec (77 specs)
evaluation/run_eval.py             one spec on one split (or a shard of it)
evaluation/postprocess.py          pooling of shards; parameter selection on validation
evaluation/paper_statistics.py     paired clip bootstrap, split statistics, size bins, residual misses
evaluation/residual_aggregate.py   pooling of residual-miss shards per configuration
evaluation/prepare_bins.py, binned_ci.py   paired intervals per area bin and per class and size
evaluation/native_pq.py            standalone evaluator for panoptic predictions (PNG + JSON)
evaluation/latency.py              runtime protocol (32 test frames per native resolution, five warm-up calls)
tracking/, tools/run_track.py, tools/eval_track.py, tools/hota.py   association configurations and metrics
tracking_eval/                     association evaluation on the test split
train.py, training/runtime.py      EoMT training (one GPU, two virtual ranks, bf16)
configs/eomt/                      EoMT configurations (eomt_large_896_joint_scratch_cp.yaml: predictor of the paper)
configs/tracking/                  the seven association configurations
configs/m2f/                       Mask2Former (Swin-S) configuration
```

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements_track.txt   # association evaluation
pip install -r requirements_m2f.txt     # Mask2Former only (MMDetection 3.3.0, mmcv 2.1.0 with CUDA ops)
```

Use Python 3.12 and install the PyTorch build that matches your CUDA version. The results of the paper were produced with Python 3.12, PyTorch 2.8 (CUDA 12.9), Lightning 2.5.1, torchmetrics 1.8.2, timm 1.0.15 and MMDetection 3.3.0. torchmetrics must be 1.8.2 (the native panoptic metric uses its internal functions), and NumPy must be below 2 (the association evaluation writes NumPy scalars to JSON).

## Data

Download MarPOT v1.0.0 from the project page and arrange it as

```text
data_panoptic_track/
├── annotations/panoptic_{train,val,test}.json
├── images/<file_name>        e.g. images/test/<sequence>/<frame>.jpg
└── panoptic/<file_name>      e.g. panoptic/test/<sequence>/<frame>.png
```

The annotations follow COCO panoptic. The sequence of a frame is the parent folder of its `file_name`; `images[]` carry `original_video_id` (the clip); `segments_info` carry `iscrowd` and `track_id` (sequence-local identity, -1: none). Categories: 1 ship, 2 obstacle (things), 3 seasurface, 4 backdrop (stuff). Panoptic PNGs encode the segment id as R + 256·G + 256²·B. The split has 5,632 / 1,402 / 2,880 frames in 256 / 70 / 158 sequences from 236 / 59 / 142 clips; frames have 2800×2800 or 2800×1600 pixels. Bootstrap intervals resample the 142 test clips (`original_video_id`), drawing all sequences of a clip together. The same `--data-root` serves training, evaluation and association.

## Checkpoints

The checkpoints of the paper are in the download folder of the project page:

| Checkpoint | Used for | File | SHA-256 |
| --- | --- | --- | --- |
| EoMT (DINOv2-L), 896-pixel input | all EoMT results | `inference_state.pt` | `850012e417a8eb8b84ba7b0c3ac0fe5f1fcf7edb4549f24e3e63aa243a01780e` |
| Mask2Former (Swin-S) | Mask2Former rows | `best_coco_panoptic_PQ_epoch_40.pth` | `d4a93ed57abea89940a9629d55ab6782571606eccb7bd55d13f8de23889add1c` |

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

Predictions follow COCO panoptic: one PNG per annotated frame at native resolution (R + 256·G + 256²·B, 0 = void) and a JSON whose `annotations[]` give `file_name` (`<sequence>/<frame>.png`, optionally prefixed by the split) or `image_id`, and `segments_info` with `id` and `category_id` (1–4). The evaluator reports PQ, PQ<sub>Th</sub>, PQ<sub>St</sub>, SQ, RQ, R<sub>S</sub>, R<sub>M</sub>, R<sub>L</sub> (native-pixel area below 32², below 96², above) and the thing TP, FP and FN; the JSON written by `--out` can be passed to `paper_statistics.py paired`. Please report the items listed on the project page: the official split, model selection on validation, input preprocessing and resolution, the number of local passes, and PQ<sub>Th</sub> and R<sub>S</sub> together with the number of thing false positives.

## Reproducing the paper

Run `python evaluation/specs.py --out configs/specs` first. Each spec is evaluated with `run_eval.py` and pooled with `postprocess.py aggregate`; paired differences use `paper_statistics.py paired` (10,000 resamples over clips, seed 0). Specs with `select_from` need `--selection` from `postprocess.py select` on the corresponding validation specs.

| Result | Specs and commands |
| --- | --- |
| Split statistics | `paper_statistics.py dataset --data DATA` |
| Test-time computation, EoMT rows | `eomt_test_{full896,full_thr03,full1344,full1792,random0,random1,random2,center,entropy,query,all}` |
| Test-time computation, Mask2Former rows | calibration grid `m2f_val_*` and `postprocess.py select`, then `m2f_test_{full,query,all}` with `--selection` and `--m2f-config configs/m2f/mask2former_swin_s_marpot.py --m2f-checkpoint best_coco_panoptic_PQ_epoch_40.pth` |
| Paired gains of query evidence | `paper_statistics.py paired --baseline <row> --candidate <query evidence>`; the three random seeds are passed together and averaged after scoring |
| Window placement and selectors | `eomt_test_{horizon,center,random0,random1,random2,entropy,density,retained}`; 560-pixel windows `eomt_test_{query_w560,center_w560,random0_w560,random1_w560,random2_w560}`; selected windows in `summary/frame_routes.jsonl` |
| Evidence ablation | `eomt_val_*` variants, then `eomt_test_retained`, `eomt_test_val_min_recall`, `eomt_test_val_max_recall` (the last two with `--selection`) |
| Window size | `eomt_test_{query,all}_w{560,840,1400}` against `eomt_test_{query,all}`; validation `eomt_val_{query,all}_w*` |
| Recall by instance size | `paper_statistics.py figures` on the shards of `eomt_test_full896` and `eomt_test_query` (list the shards explicitly; a glob also matches the window variants); intervals with `prepare_bins.py` and `binned_ci.py` |
| Runtimes and stage breakdown | `evaluation/latency.py`, see below |
| Association | `eomt_test_full896` and `eomt_test_query`, then `tracking_eval/` (see below) |
| Residual misses | `eomt_val_query`, then `paper_statistics.py residual --data DATA --inputs <shard summaries>` (the split is taken from the inputs) and `residual_aggregate.py` |
| Validation studies (placement, budget, oracle inputs) | `eomt_val_*`; variant selection with `postprocess.py select` |
| EoMT training runs | `train.py --config configs/eomt/<config>.yaml --data-root DATA --out OUT` |

### Runtime

```bash
python evaluation/latency.py --data-root DATA --config configs/eomt/eomt_large_896_joint_scratch_cp.yaml \
  --checkpoint inference_state.pt --specs-dir configs/specs --out runs/latency [--stages] \
  [--specs ... m2f_test_full m2f_test_query --m2f-config configs/m2f/mask2former_swin_s_marpot.py \
   --m2f-checkpoint best_coco_panoptic_PQ_epoch_40.pth --selection SELECTION.json]
```

The frame pool is fixed before any prediction: the longest contiguous run of each native resolution in the test annotations, 32 frames each. Each spec is run once per resolution on an otherwise idle GPU with batch size 1; the prediction call (resize, forward passes, window selection and composition) is timed with the GPU synchronized, after five warm-up calls. `summary.json` gives the mean per resolution and the mean weighted by the number of test frames of each resolution; `--stages` adds the breakdown into preprocessing, forward passes, window selection, composition and other.

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

With sharded runs, pass the shard summaries of exactly one spec to each `build_cache_manifest.py` call. The tracker implementations follow ByteTrack, OC-SORT and BoT-SORT; HOTA is computed with TrackEval.

### Training

```bash
python train.py --config configs/eomt/eomt_large_896_joint_scratch_cp.yaml --data-root DATA --out runs/train_eomt_896 \
  --init-checkpoint weights/coco_panoptic_eomt_large_640/pytorch_model.bin
```

Training runs on one GPU in bf16; configurations written for two devices are emulated with two virtual ranks (physical batch 2). It selects the checkpoint with the best validation PQ; the attention-mask annealing steps are recomputed from the actual number of steps (2,816 steps per epoch and 140,800 updates for the main configuration). Add `--resume-own` to resume a run. The warm-start configuration `eomt_large_640_joint.yaml` needs `--init-checkpoint` pointing to the `inference_state.pt` of a panoptic-only 640 run (`eomt_large_640_track.yaml`). The output `inference_state.pt` is the checkpoint used by `run_eval.py`.

## Not included

Baseline training code and third-party code bases (MMSegmentation, Detectron2, PanSR, PEM, FA-PEM, DAPE, SegMAN, GCNet), Mask2Former training (the configuration is used for inference), the PanSR and LaRS adaptations, query evidence on 1344- and 1792-pixel full-frame passes, the near-duplicate analysis, figure scripts and job orchestration.

## License

MIT (see `LICENSE` and `NOTICE`). Files taken from EoMT keep their copyright headers; portions adapted from Apache-2.0 projects are listed in `NOTICE`. The dataset is released under the MarPOT Terms of Use on the project page.

## Citation

```bibtex
@misc{marpot2026,
  title        = {{MarPOT}: A High-Resolution Maritime Panoptic Benchmark and a Training-Free Re-Observation Baseline for Small Instances},
  year         = {2026},
  howpublished = {\url{https://marpot-benchmark.github.io/}},
  note         = {Dataset version 1.0.0}
}
```
