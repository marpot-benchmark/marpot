#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
from run_eval import dump,sha
from evaluation_identity import validate_shard_identities


def aggregate(a):
    sys.path.insert(0,a.root)
    from evaluation.combine import combine
    specs=json.loads(Path(a.spec).read_text())
    inputs=[Path(x) for x in a.inputs]
    parts=[json.loads(f.read_text()) for f in inputs]
    identity,identity_fingerprint=validate_shard_identities(parts,specs)
    ranges=sorted(part['index_range'] for part in parts)
    end=0
    for lo,hi in ranges:
        if lo!=end or hi<=lo:raise ValueError('shard overlap or gap')
        end=hi
    if end!=parts[0]['split_images']:raise ValueError('missing final shard')
    job=dict(specs,experiment=specs['family']);check=dict(job,selector='all')
    result=combine(inputs,Path(a.out)/'summary.json',check)
    result['job']=job
    result['evaluation_identity']=identity
    result['evaluation_fingerprint']=identity_fingerprint
    if specs['selector'] not in ('full','all'):
        expected=result['images']*specs['k']
        if result['selected_tiles']!=expected:raise ValueError(f'budget {result["selected_tiles"]} != {expected}')
    result['input_hashes']={str(f):sha(f) for f in inputs}
    import numpy as np
    from paper_statistics import load_stats,metrics
    native=sum((load_stats(f)[1].sum(0) for f in inputs),np.zeros((2,7)))
    pq,recall=metrics(native)
    if abs(float(pq)-result['pq']['things']['pq'])>1e-6:raise ValueError('Native sufficient statistics do not reconstruct aggregate PQ')
    result['native_metrics']=dict(PQ_Th=float(pq),R_S=float(recall),Small_TP=int(native[:,4].sum()),
        Small_GT=int(native[:,4:6].sum()),units='fraction',
        definition='Native GT-void-aware matches; fixed two-thing-class PQ; small GT area < 1024 pixels')
    dump(Path(a.out)/'summary.json',result)


def select(a):
    records=[]
    for f in map(Path,a.inputs):
        r=json.loads(f.read_text())
        if r['job']['split']!='val':raise ValueError('test outputs cannot select parameters')
        records.append((f,r))
    if not records:raise ValueError('no validation candidates')
    selectors={}
    for group in sorted({r['job']['options'].get('calibration_group') for _,r in records}-{None}):
        items=[(f,r) for f,r in records if r['job']['options'].get('calibration_group')==group]
        rules=['min_small_recall','max_small_recall'] if group=='eomt_evidence_variants' else ['max_pq_then_recall']
        for rule in rules:
            if rule=='min_small_recall':f,r=min(items,key=lambda x:(x[1]['small_recall'],x[1]['pq']['things']['pq'],x[1]['job']['id']))
            elif rule=='max_small_recall':f,r=max(items,key=lambda x:(x[1]['small_recall'],x[1]['pq']['things']['pq'],x[1]['job']['id']))
            else:f,r=max(items,key=lambda x:(x[1]['pq']['things']['pq'],x[1]['small_recall'],x[1]['job']['id']))
            selectors[group+':'+rule]=dict(split='val',source=str(f),source_sha256=sha(f),
                options=r['job']['options'],candidate_id=r['job']['id'],selection_rule=rule)
    dump(Path(a.out)/'selection.json',dict(selections=selectors,
         all_candidate_hashes={str(f):sha(f) for f,r in records}))


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['aggregate','select'])
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]),help='repository root (default: parent of evaluation/)')
    p.add_argument('--out',required=True)
    p.add_argument('--spec');p.add_argument('--inputs',nargs='+',required=True)
    a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True)
    (aggregate if a.operation=='aggregate' else select)(a)


if __name__=='__main__':main()
