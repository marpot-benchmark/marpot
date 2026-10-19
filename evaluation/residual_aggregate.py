#!/usr/bin/env python3
import argparse,collections,json
from pathlib import Path
from run_eval import dump,sha


def aggregate(inputs,data,split='val'):
    counts=collections.Counter();failures=[];ranges=[];sources={};seen=set()
    for path in map(Path,inputs):
        record=json.loads(path.read_text());sources[str(path)]=sha(path)
        local_frames=0
        for origin,digest in record['input_hashes'].items():
            origin=Path(origin)
            if sha(origin)!=digest:raise ValueError('residual input changed')
            shard=json.loads(origin.read_text())
            if str(origin.resolve()) in seen:raise ValueError('duplicate residual source')
            seen.add(str(origin.resolve()));ranges.append(shard['index_range'])
            local_frames+=shard['images']
        if local_frames!=record['frames']:raise ValueError('residual frame count mismatch')
        if sum(record['counts'].values())!=len(record['failures']):raise ValueError('residual failure count mismatch')
        counts.update(record['counts']);failures.extend(record['failures'])
    end=0
    for lo,hi in sorted(ranges):
        if lo!=end or hi<=lo:raise ValueError('residual validation coverage has a gap or overlap')
        end=hi
    expected=len(json.loads((Path(data)/f'annotations/panoptic_{split}.json').read_text())['images'])
    if end!=expected:raise ValueError('residual validation coverage incomplete')
    return dict(split=split,frames=end,counts=dict(counts),total_small_false_negatives=len(failures),
        failures=failures,input_hashes=sources,
        definition='First blocking stage for ordinary same-class mask-IoU > 0.5 recovery; GT area < 1024 pixels. Native void-aware PQ is a separate channel.')


def aggregate_conditions(inputs, data, split='val'):
    groups = {}
    metadata = {}
    population = None
    for path in map(Path, inputs):
        record = json.loads(path.read_text())
        condition = None
        for source, expected_hash in record['input_hashes'].items():
            if sha(source) != expected_hash:
                raise ValueError('residual input changed')
            shard = json.loads(Path(source).read_text())
            identity = shard['evaluation_identity']
            spec = dict(identity['spec'])
            spec['options'] = {k: v for k, v in spec['options'].items()
                               if k not in {'trace', 'raw_trace', 'save_candidates', 'tracking'}}
            key = json.dumps(spec, sort_keys=True)
            if condition is not None and condition != key:
                raise ValueError('one residual input mixes prediction conditions')
            condition = key
            current_population = dict(data=identity['data'], checkpoints=identity['checkpoints'],
                                      config_sha256=identity['config_sha256'])
            if population is not None and population != current_population:
                raise ValueError('residual conditions use different data or checkpoints')
            population = current_population
            metadata[key] = spec
        if condition is None:
            raise ValueError('residual input has no provenance')
        groups.setdefault(condition, []).append(path)
    if not groups:
        raise ValueError('no residual inputs')
    if len(groups) == 1:
        return aggregate(next(iter(groups.values())), data, split)
    variants = {}
    primary = None
    for key, paths in groups.items():
        spec = metadata[key]
        name = spec['id']
        if name in variants:
            raise ValueError('same condition name has conflicting settings')
        variants[name] = dict(condition=spec, **aggregate(paths, data, split))
        if name == f'eomt_{split}_query' and spec['window'] == 1120 and spec['selector'] == 'query-small' and spec['k'] == 3:
            primary = name
    if primary is None:
        raise ValueError('default 1120-pixel three-window query condition missing')
    result = {k: v for k, v in variants[primary].items() if k != 'condition'}
    result.update(primary_condition=primary, variants=variants,
                  input_hashes={str(p): sha(p) for p in map(Path, inputs)},
                  aggregation='Separate prediction conditions; pool only disjoint shards within each condition.',
                  unique_validation_frames=result['frames'],
                  condition_count=len(variants),
                  repair_note='Manifest prefix selected four window-size conditions, not four disjoint validation shards. Primary counts retain the default 1120-pixel query setting.')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--split',default='val',choices=['val','test'])
    a=p.parse_args()
    out=Path(a.out);dump(out/'summary.json',aggregate_conditions(a.inputs,a.data,a.split))


if __name__=='__main__':main()
