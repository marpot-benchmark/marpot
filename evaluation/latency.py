#!/usr/bin/env python3
import argparse, ast, contextlib, functools, hashlib, inspect, json, os
import subprocess, sys, textwrap, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_SPECS = ['eomt_test_full896', 'eomt_test_full1344', 'eomt_test_full1792', 'eomt_test_query',
                 'eomt_test_horizon', 'eomt_test_all', 'eomt_test_center', 'eomt_test_random0',
                 'eomt_test_density', 'eomt_test_entropy',
                 'eomt_test_query_w560', 'eomt_test_all_w560', 'eomt_test_query_w840', 'eomt_test_all_w840',
                 'eomt_test_query_w1400', 'eomt_test_all_w1400']
STAGE_SPECS = ['eomt_test_full896', 'eomt_test_full1344', 'eomt_test_query']


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def shape_intervals(images, samples):
    groups = {}; start = 0
    for i in range(1, len(images) + 1):
        previous = images[i - 1]
        if i == len(images) or (images[i]['height'], images[i]['width']) != (previous['height'], previous['width']):
            groups.setdefault((previous['height'], previous['width']), []).append((start, i)); start = i
    return [dict(shape=list(shape), start=lo, end=min(lo + samples, hi)) for shape, runs in sorted(groups.items())
            for lo, hi in [max(runs, key=lambda x: (x[1] - x[0], -x[0]))]]


class StageTimers:
    def __init__(self, model): self.model = model; self.undo = []; self.sums = {}; self.active = False
    def reset(self): self.sums = {}; self.active = True
    @contextlib.contextmanager
    def section(self, name):
        import torch
        if not self.active: yield; return
        torch.cuda.synchronize(); start = time.perf_counter()
        try: yield
        finally:
            torch.cuda.synchronize(); self.sums[name] = self.sums.get(name, 0.) + time.perf_counter() - start
    def wrap(self, obj, name, bucket):
        old = getattr(obj, name)
        @functools.wraps(old)
        def call(*a, **kw):
            with self.section(bucket): return old(*a, **kw)
        self.undo.append((obj, name, old)); setattr(obj, name, call)
    def split_forward(self, obj, first_bucket, rest_bucket):
        old = obj.forward; tree = ast.parse(textwrap.dedent(inspect.getsource(old)))
        fn = tree.body[0]; fn.decorator_list = []
        def wrap(stmts, bucket):
            return ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='_latency_stages', ctx=ast.Load()), attr='section', ctx=ast.Load()), args=[ast.Constant(bucket)], keywords=[]), optional_vars=None)], body=stmts)
        fn.body = [wrap(fn.body[:1], first_bucket)] + ([wrap(fn.body[1:], rest_bucket)] if rest_bucket else fn.body[1:])
        ast.fix_missing_locations(tree); ns = {**old.__func__.__globals__, '_latency_stages': self}
        exec(compile(tree, '<latency_stages>', 'exec'), ns)
        self.undo.append((obj, 'forward', old)); obj.forward = types.MethodType(ns[fn.name], obj)
    def __enter__(self):
        import tools.eomt_infer as e, tools.tiled_panoptic_infer as t, tools.reobs_core as core
        self.wrap(self.model, 'resize_and_pad_imgs_instance_panoptic', 'pre')
        self.split_forward(self.model, 'pre', None)
        self.split_forward(self.model.network, 'pre', 'fwd')
        self.wrap(e, 'query_routing_evidence', 'sel'); self.wrap(core, 'select_supplement_tiles', 'sel')
        for n in ['merge_same_class_instances', 'gate_tile_detections', 'compose_panoptic']: self.wrap(t, n, 'comp')
        return self
    def __exit__(self, *exc):
        self.active = False
        for obj, name, old in reversed(self.undo): setattr(obj, name, old)


def child(argv):
    p = argparse.ArgumentParser(); p.add_argument('--latency-out', required=True)
    p.add_argument('--warmup', type=int, default=5); p.add_argument('--stages', action='store_true')
    a, forward = p.parse_known_args(argv)
    sys.path[:0] = [str(ROOT), str(HERE)]
    import numpy as np, torch
    import tools.analyze_panoptic_errors as evaluator
    from run_eval import main as evaluate
    original = evaluator.predict; times = []; stages = []; gate = {}
    def timed(*args, **kwargs):
        torch.cuda.synchronize(); start = time.perf_counter(); value = original(*args, **kwargs); torch.cuda.synchronize()
        return value, time.perf_counter() - start
    def predict(*args, **kwargs):
        if not times:
            for _ in range(a.warmup): original(*args, **kwargs)
        if not a.stages:
            value, seconds = timed(*args, **kwargs); times.append(seconds); return value
        with StageTimers(args[0]) as st:
            if not times:
                reference = original(*args, **kwargs)[0].cpu().numpy().tobytes()
                st.reset(); check = original(*args, **kwargs)[0].cpu().numpy().tobytes()
                gate['instrumented_prediction_identical'] = reference == check
                if reference != check: raise RuntimeError('stage instrumentation changed the prediction')
            st.reset(); value, seconds = timed(*args, **kwargs)
        times.append(seconds)
        row = {k: st.sums.get(k, 0.) for k in ['pre', 'fwd', 'sel', 'comp']}
        row['total'] = seconds; row['other'] = seconds - sum(row[k] for k in ['pre', 'fwd', 'sel', 'comp'])
        stages.append(row); return value
    evaluator.predict = predict
    sys.argv = ['run_eval.py', *forward]; evaluate()
    result = dict(images=len(times), seconds=times, warmup=a.warmup, mean_seconds=float(np.mean(times)),
                  device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda)
    if a.stages:
        result['stages_ms'] = {k: float(np.mean([r[k] for r in stages])) * 1000 for k in ['pre', 'fwd', 'sel', 'comp', 'other', 'total']}
        result.update(gate)
    dump(Path(a.latency_out), result)


def main():
    p = argparse.ArgumentParser(description='Runtime protocol of the paper: 32 test frames per native resolution, five warm-up calls, batch 1.')
    p.add_argument('--data-root', required=True, help='data_panoptic_track directory')
    p.add_argument('--config', required=True, help='EoMT config, e.g. configs/eomt/eomt_large_896_joint_scratch_cp.yaml')
    p.add_argument('--checkpoint', required=True, help='EoMT checkpoint')
    p.add_argument('--specs-dir', default=str(ROOT / 'configs/specs'), help='directory written by evaluation/specs.py --out')
    p.add_argument('--specs', nargs='+', help='spec ids to time (default: the EoMT runtime rows of the paper; add m2f_test_full m2f_test_query explicitly)')
    p.add_argument('--selection', help='validation selection JSON (postprocess.py select), needed for m2f_test_query')
    p.add_argument('--m2f-config'); p.add_argument('--m2f-checkpoint')
    p.add_argument('--samples', type=int, default=32); p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--stages', action='store_true', help='per-stage breakdown of full896, full1344 and query')
    p.add_argument('--out', required=True)
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    images = json.loads((Path(a.data_root) / 'annotations/panoptic_test.json').read_text())['images']
    counts = {}
    for image in images: counts[(image['height'], image['width'])] = counts.get((image['height'], image['width']), 0) + 1
    intervals = shape_intervals(images, a.samples)
    dump(out / 'image_pool.json', dict(split='test', intervals=intervals,
         files=[images[i]['file_name'] for x in intervals for i in range(x['start'], x['end'])]))
    names = a.specs or DEFAULT_SPECS
    runs = [(name, False) for name in names] + ([(name, True) for name in STAGE_SPECS] if a.stages else [])
    rows = []
    for name, stages in runs:
        spec = json.loads((Path(a.specs_dir) / (name + '.json')).read_text())
        spec['split'] = 'test'
        spec['options'].update(tracking=False, save_candidates=False, raw_trace=False, trace=False)
        spec_file = out / 'specs' / (name + '.json'); dump(spec_file, spec)
        per_shape = []
        for interval in intervals:
            tag = name + ('_stages_' if stages else '_') + 'x'.join(map(str, interval['shape']))
            folder = out / 'runs' / tag
            cmd = [sys.executable, str(Path(__file__).resolve()), '--child', '--latency-out', str(folder / 'latency.json'),
                   '--warmup', str(a.warmup), *(['--stages'] if stages else []),
                   '--data-root', a.data_root, '--config', a.config, '--checkpoint', a.checkpoint,
                   '--spec', str(spec_file), '--out', str(folder), '--start', str(interval['start']), '--end', str(interval['end'])]
            if name.startswith('m2f_'):
                if not a.m2f_config or not a.m2f_checkpoint: raise ValueError('Mask2Former specs need --m2f-config and --m2f-checkpoint')
                cmd += ['--m2f-config', a.m2f_config, '--m2f-checkpoint', a.m2f_checkpoint]
            if spec['options'].get('select_from'):
                if not a.selection: raise ValueError(name + ' needs --selection')
                cmd += ['--selection', a.selection]
            subprocess.run(cmd, check=True)
            result = json.loads((folder / 'latency.json').read_text())
            per_shape.append(dict(shape=interval['shape'], test_frames=counts[tuple(interval['shape'])], **result))
            print(f"{tag:40s} {1000 * result['mean_seconds']:9.1f} ms", flush=True)
        total = sum(x['test_frames'] for x in per_shape)
        row = dict(spec=name, stages=stages, per_shape=per_shape,
                   weighted_mean_ms=1000 * sum(x['test_frames'] * x['mean_seconds'] for x in per_shape) / total)
        if stages:
            row['weighted_stages_ms'] = {k: sum(x['test_frames'] * x['stages_ms'][k] for x in per_shape) / total
                                         for k in ['pre', 'fwd', 'sel', 'comp', 'other', 'total']}
        rows.append(row); dump(out / 'summary.json', dict(image_pool=intervals, cases=rows))
    for row in rows:
        print(f"{row['spec'] + (' (stages)' if row['stages'] else ''):40s} weighted {row['weighted_mean_ms']:9.1f} ms")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        child(sys.argv[2:])
    else:
        main()
