import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import yaml


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def export_inference_checkpoint(selected, out):
    out = Path(out)
    source = Path(selected['path'])
    if sha256(source) != selected['sha256']:
        raise ValueError('selected training checkpoint changed before export')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor)
               for k, v in state.items()):
        raise TypeError('unexpected non-tensor model state')
    path = out / 'inference_state.pt'
    tmp = path.with_suffix('.pt.tmp')
    torch.save(dict(state_dict=state, epoch=int(checkpoint['epoch']),
                    global_step=int(checkpoint['global_step']),
                    metadata=dict(training_checkpoint_sha256=selected['sha256'])), tmp)
    os.replace(tmp, path)
    slim = torch.load(path, map_location='cpu', weights_only=True)
    if list(slim['state_dict']) != list(state) or any(
            not torch.equal(slim['state_dict'][k], value) for k, value in state.items()):
        raise RuntimeError('inference export state differs from selected checkpoint')
    result = dict(selected, path=str(path.resolve()), sha256=sha256(path),
                  training_checkpoint_path=str(source.resolve()),
                  training_checkpoint_sha256=selected['sha256'],
                  format='tensor_state_dict', weights_only_load_verified=True)
    atomic_json(out / 'inference_selected.json', result)
    return result


def instantiate(spec):
    if isinstance(spec, dict) and 'class_path' in spec:
        mod, name = spec['class_path'].rsplit('.', 1)
        return getattr(importlib.import_module(mod), name)(
            **{k: instantiate(v) for k, v in spec.get('init_args', {}).items()})
    return spec


def slice_outputs(value, start, end):
    if isinstance(value, torch.Tensor):
        return value[start:end]
    if isinstance(value, tuple):
        return tuple(slice_outputs(x, start, end) for x in value)
    if isinstance(value, list):
        return [slice_outputs(x, start, end) for x in value]
    if isinstance(value, dict):
        return {k: slice_outputs(v, start, end) for k, v in value.items()}
    return value


class VirtualRankSampler(Sampler):
    def __init__(self, dataset, ranks, rank_batch, seed=0):
        self.dataset=dataset;self.ranks=ranks;self.rank_batch=rank_batch
        self.seed=seed;self.epoch=0
        rank_samples=(len(dataset)+ranks-1)//ranks
        self.rank_used=rank_samples//rank_batch*rank_batch

    def __len__(self):return self.rank_used*self.ranks

    def set_epoch(self,epoch):self.epoch=epoch

    def __iter__(self):
        indices=[]
        for rank in range(self.ranks):
            sampler=DistributedSampler(self.dataset,num_replicas=self.ranks,
                rank=rank,shuffle=True,seed=self.seed,drop_last=False)
            sampler.set_epoch(self.epoch);indices.append(list(sampler))
        for offset in range(0,self.rank_used,self.rank_batch):
            for rank in range(self.ranks):
                yield from indices[rank][offset:offset+self.rank_batch]


def virtual_rank_training_step(self, batch, batch_idx):
    imgs, targets = batch
    g = self.virtual_shards
    if len(targets) % g:
        raise ValueError('physical batch must divide into equal virtual ranks')
    if self.trainer.world_size != 1 or self.trainer.accumulate_grad_batches != 1:
        raise ValueError('virtual ranks require one real GPU and no accumulation')
    n = len(targets) // g
    outputs = self(imgs)
    avg = {}
    assoc = outputs[0][-1].new_zeros(())
    for r in range(g):
        lo, hi = r * n, (r + 1) * n
        sub = slice_outputs(outputs, lo, hi)
        tgts = targets[lo:hi]
        for i, (masks, classes) in enumerate(zip(sub[0], sub[1])):
            loss = self.criterion(masks_queries_logits=masks,
                                  class_queries_logits=classes, targets=tgts)
            q = masks.shape[1]
            local = sum(min(q, len(t['labels'])) for t in tgts)
            total = sum(min(q, len(t['labels'])) for t in targets)
            correction = max(local, 1) / max(total / g, 1)
            for k, value in loss.items():
                if k in ('loss_mask', 'loss_dice'):
                    value = value * correction
                key = k + self.block_postfix(i)
                avg[key] = avg.get(key, 0) + value / g
        if self.assoc_coefficient > 0:
            a = self._association_loss(imgs[lo:hi], tgts, sub)
            if a is not None:
                assoc = assoc + a / g
    total_loss = self.criterion.loss_total(avg, self.log)
    if self.assoc_coefficient > 0:
        self.log('losses/train_loss_assoc', assoc, sync_dist=False)
        total_loss = total_loss + self.assoc_coefficient * assoc
    return total_loss


def final_eval_step(self, batch, batch_idx=None, log_prefix=None):
    imgs, targets = batch
    sizes = [self._cap_eval_size(img.shape[-2:]) for img in imgs]
    outputs = self(self.resize_and_pad_imgs_instance_panoptic(imgs))
    crowds = [t['is_crowd'] for t in targets]
    target_maps = self.to_per_pixel_targets_panoptic(targets)
    if self.eval_max_size is not None:
        target_maps = [self._resize_per_pixel_target(t, s)
                       for t, s in zip(target_maps, sizes)]
    masks = F.interpolate(outputs[0][-1], self.img_size, mode='bilinear')
    masks = self.revert_resize_and_pad_logits_instance_panoptic(masks, sizes)
    preds = self.to_per_pixel_preds_panoptic(
        masks, outputs[1][-1], self.stuff_classes, self.mask_thresh,
        self.overlap_thresh)
    self.update_metrics_panoptic(preds, target_maps, crowds, len(self.metrics) - 1)


def final_epoch_end(self):
    metric = self.metrics[-1]
    result = metric.compute()[:-1]
    metric.reset()
    n = len(metric.things)
    for j, key in enumerate(('pq', 'sq', 'rq')):
        values = result[:, j]
        self.log(f'metrics/val_{key}_all', values.mean())
        self.log(f'metrics/val_{key}_things', values[:n].mean())
        self.log(f'metrics/val_{key}_stuff', values[n:].mean())
        for c, value in enumerate(values):
            self.log(f'metrics/val_{key}_class_{c}', value)


def build(root, recipe, data, batch_size=None, workers=4,
          checkpointing=False, virtual_shards=0, final_only=False,
          init_checkpoint=None):
    from types import MethodType
    cfg = yaml.safe_load(Path(recipe).read_text())
    d = cfg['data']['init_args'].copy()
    module, cls_name = cfg['data']['class_path'].rsplit('.', 1)
    data_cls = getattr(importlib.import_module(module), cls_name)
    for key in ('img_size', 'num_classes', 'batch_size'):
        if key not in d:
            default = inspect.signature(data_cls.__init__).parameters[key].default
            if default is inspect.Parameter.empty:
                raise ValueError('missing required dataset parameter: ' + key)
            d[key] = default
    original_global = d['batch_size'] * cfg['trainer'].get('devices', 1)
    d.update(path=str(data), num_workers=workers)
    if batch_size is not None:
        d['batch_size'] = batch_size
    init = Path(init_checkpoint or
                Path(root) / 'weights/coco_panoptic_eomt_large_640/pytorch_model.bin')
    if not init.is_file():
        raise FileNotFoundError(init)
    m = cfg['model']['init_args'].copy()
    m.update(img_size=d['img_size'], num_classes=d['num_classes'],
             stuff_classes=d['stuff_classes'], ckpt_path=str(init))
    net = m['network']
    net['init_args'].update(num_classes=d['num_classes'],
                             grad_checkpointing=checkpointing)
    net['init_args']['encoder']['init_args'] = dict(
        net['init_args']['encoder'].get('init_args', {}),
        img_size=d['img_size'], ckpt_path=str(init))
    m['network'] = instantiate(net)
    model = instantiate(dict(class_path=cfg['model']['class_path'], init_args=m))
    dm = instantiate(dict(class_path=cfg['data']['class_path'], init_args=d))
    if workers:
        dm.dataloader_kwargs['prefetch_factor'] = 2
    if virtual_shards:
        if d['batch_size'] % virtual_shards:
            raise ValueError('physical batch must divide into virtual ranks')
        model.virtual_shards = virtual_shards
        model.training_step = MethodType(virtual_rank_training_step, model)
        original_loader = dm.train_dataloader
        def train_dataloader(self):
            loader = original_loader()
            sampler = VirtualRankSampler(loader.dataset,virtual_shards,
                d['batch_size']//virtual_shards,int(os.environ.get('PL_GLOBAL_SEED','0')))
            return DataLoader(loader.dataset,**self.dataloader_kwargs,
                sampler=sampler,drop_last=True,collate_fn=loader.collate_fn)
        dm.train_dataloader = MethodType(train_dataloader,dm)
    if final_only:
        model.eval_step = MethodType(final_eval_step, model)
        model.on_validation_epoch_end = MethodType(final_epoch_end, model)
    metadata = dict(recipe=str(recipe), recipe_sha256=sha256(recipe),
                    data=str(data), init_checkpoint=str(init),
                    original_global_batch=original_global,
                    physical_batch=d['batch_size'], virtual_shards=virtual_shards,
                    grad_checkpointing=checkpointing, final_only_validation=final_only,
                    dataloader=dict(dm.dataloader_kwargs),
                    sampler='virtual_distributed_padding_and_rank_drop_last' if virtual_shards else 'original_random_sampler',
                    loss_semantics='virtual_rank_mean' if virtual_shards else 'native_batch_mean')
    return model, dm, cfg, metadata
