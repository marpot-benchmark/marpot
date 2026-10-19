import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger


def scalar_config(value):
    if isinstance(value, torch.nn.Module):
        cls = type(value)
        return {'omitted_module': cls.__module__ + '.' + cls.__qualname__}
    if isinstance(value, torch.Tensor):
        return {'omitted_tensor': True, 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return scalar_config(vars(value))
    if isinstance(value, dict):
        return {str(k): scalar_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_config(v) for v in value]
    cls = type(value)
    return {'omitted_object': cls.__module__ + '.' + cls.__qualname__}


class SafeCSVLogger(CSVLogger):
    def log_hyperparams(self, params=None):
        return super().log_hyperparams(scalar_config(params or {}))


def prepare_safe_logging(model, metadata, attempt):
    model._log_hyperparams = False
    logger = SafeCSVLogger(str(attempt), name='metrics', version=0,
                           flush_logs_every_n_steps=20)
    logger.log_hyperparams(metadata)
    return logger


class TrainingProgress(Callback):
    def __init__(self, attempt, interval=20):
        self.attempt = Path(attempt)
        self.interval = interval
        self.started = time.monotonic()
        self.previous_step = None
        self.first = True

    def on_train_start(self, trainer, pl_module):
        self.previous_step = int(trainer.global_step)
        self.started = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        advanced = self.previous_step is not None and step > self.previous_step
        self.previous_step = step
        if not advanced or (not self.first and step % self.interval):
            return
        self.first = False
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        loss_value = float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else float(loss)
        report = dict(event='training_progress', global_step=step,
                      optimizer_step_completed=True, epoch=int(trainer.current_epoch),
                      batch_idx=int(batch_idx), loss=loss_value if math.isfinite(loss_value) else None,
                      finite_loss=math.isfinite(loss_value),
                      elapsed_seconds=time.monotonic()-self.started, time=time.time())
        self.attempt.mkdir(parents=True, exist_ok=True)
        line = json.dumps(report, sort_keys=True, allow_nan=False)
        with (self.attempt/'training_progress.jsonl').open('a') as f:
            f.write(line+'\n')
        tmp = self.attempt/'training_progress.json.tmp'
        tmp.write_text(line+'\n')
        os.replace(tmp, self.attempt/'training_progress.json')
        print(line, flush=True)
