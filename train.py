#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
p=argparse.ArgumentParser()
p.add_argument('--config',dest='recipe',metavar='CONFIG',required=True,
               help='EoMT recipe, e.g. configs/eomt/eomt_large_896_joint_scratch_cp.yaml')
p.add_argument('--data-root',dest='data',metavar='DATA_ROOT',required=True,
               help='MarPOT data root (annotations/, images/, panoptic/); replaces data.init_args.path')
p.add_argument('--out',required=True)
p.add_argument('--init-checkpoint',default=None,
               help='COCO panoptic EoMT-L 640 weights (default: weights/coco_panoptic_eomt_large_640/'
                    'pytorch_model.bin under the repository root); the warm-start recipe '
                    '(load_ckpt_class_head: True) requires its trained predecessor checkpoint here')
p.add_argument('--batch',type=int,default=None)
p.add_argument('--virtual-shards',type=int,default=None)
p.add_argument('--workers',type=int,default=8)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--checkpointing',action='store_true')
p.add_argument('--resume-own',action='store_true')
p.add_argument('--all-heads-validation',action='store_true')
a=p.parse_args()
from training.runtime import atomic_json,build,sha256,export_inference_checkpoint
from training.resume_state import plan_resume,begin_attempt
import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from training.logging_safe import prepare_safe_logging,TrainingProgress

out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
cfg=yaml.safe_load(Path(a.recipe).read_text())
old_devices=int(cfg['trainer'].get('devices',1))
expected_batch=old_devices*int(cfg['data']['init_args']['batch_size'])
batch=a.batch or expected_batch
virtual=old_devices if a.virtual_shards is None else a.virtual_shards
if virtual==1:virtual=0
if batch!=expected_batch:
    raise ValueError(f'physical batch {batch} changes frozen global batch {expected_batch}')
if cfg['model']['init_args'].get('load_ckpt_class_head',False) and not a.init_checkpoint:
    raise ValueError('warm-start recipe requires the NEW predecessor checkpoint explicitly')
manifest=out/'run_manifest.json'
init_path=Path(a.init_checkpoint or ROOT/'weights/coco_panoptic_eomt_large_640/pytorch_model.bin').resolve()
identity=dict(recipe=str(a.recipe),recipe_sha256=sha256(a.recipe),data=str(a.data),
              init_checkpoint=str(init_path),init_checkpoint_sha256=sha256(init_path),
              physical_batch=batch,effective_batch=batch,virtual_shards=virtual,
              grad_checkpointing=a.checkpointing,final_only_validation=not a.all_heads_validation,
              precision='bf16-mixed',seed=a.seed,loader_workers=a.workers)
resume_plan=plan_resume(out,identity,a.resume_own)
pl.seed_everything(a.seed,workers=True)
torch.set_float32_matmul_precision('medium')
torch.set_num_threads(min(4,os.cpu_count() or 1))
torch.backends.cuda.enable_cudnn_sdp(False)
model,dm,cfg,metadata=build(ROOT,a.recipe,a.data,batch,a.workers,
                           a.checkpointing,virtual,not a.all_heads_validation,
                           a.init_checkpoint)
dm.setup('fit')
steps_per_epoch=len(dm.train_dataloader())
epochs=int(cfg['trainer']['max_epochs'])
if model.attn_mask_annealing_enabled:
    total=steps_per_epoch*epochs
    model.attn_mask_annealing_start_steps=[round(total*(i+1)/6) for i in range(4)]
    model.attn_mask_annealing_end_steps=[round(total*(i+2)/6) for i in range(4)]
metadata.update(epochs=epochs,steps_per_epoch=steps_per_epoch,
                train_samples=len(dm.train_dataset),val_samples=len(dm.val_dataset),
                effective_batch=batch,precision='bf16-mixed',seed=a.seed,
                cudnn_sdpa_enabled=False,
                init_checkpoint_sha256=sha256(metadata['init_checkpoint']),
                hardware=torch.cuda.get_device_name(0),
                annealing_start=model.attn_mask_annealing_start_steps,
                annealing_end=model.attn_mask_annealing_end_steps,
                started_at=time.time(),
                equivalence_boundary='matches historical virtual-rank loss reductions; stochastic samples and batched CUDA numerics are not bitwise DDP replay')
metadata.update(identity)
if not manifest.exists():atomic_json(manifest,metadata)
attempt=begin_attempt(out,identity,resume_plan)
cb=ModelCheckpoint(dirpath=out/'checkpoints',monitor='metrics/val_pq_all',
                   mode='max',save_top_k=1,save_last=True,
                   filename='best-{epoch:03d}-{step:06d}')
class FiniteGradientGate(Callback):
    def on_before_optimizer_step(self, trainer, module, optimizer):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        if not torch.isfinite(torch.stack(torch._foreach_norm(grads))).all():
            raise RuntimeError('nonfinite gradient: aborting before optimizer mutation')
logger=prepare_safe_logging(model,metadata,attempt)
trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',
                   max_epochs=epochs,logger=logger,
                   callbacks=[cb,FiniteGradientGate(),TrainingProgress(attempt)],num_sanity_val_steps=0,
                   enable_progress_bar=False,enable_model_summary=False,
                   gradient_clip_val=.01,gradient_clip_algorithm='norm',
                   log_every_n_steps=50)
trainer.fit(model,datamodule=dm,ckpt_path=resume_plan['checkpoint'])
if trainer.current_epoch!=epochs or not cb.best_model_path or cb.best_model_score is None:
    raise RuntimeError('incomplete training or absent validation-selected checkpoint')
best=Path(cb.best_model_path).resolve()
selected=dict(path=str(best),sha256=sha256(best),
              validation_score=float(cb.best_model_score),
              validation_metric='metrics/val_pq_all',
              validation_resolution_cap=model.eval_max_size,
              precision='bf16-mixed',
              physical_batch=batch,effective_batch=batch,
              loss_semantics=metadata['loss_semantics'],virtual_shards=virtual,
              completed_epochs=trainer.current_epoch,
              completed_steps=trainer.global_step,
              training_attempt=str(attempt),
              restart_reason=resume_plan['restart_reason'])
atomic_json(out/'selected_checkpoint.json',selected)
inference_selected=export_inference_checkpoint(selected,out)
print(json.dumps(selected))
