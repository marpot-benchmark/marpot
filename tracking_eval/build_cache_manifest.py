#!/usr/bin/env python3
import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tracking_eval.common import dataset_rows,digest,dump,load_data,read,rows,sha,validate_frame

def main():
    p=argparse.ArgumentParser();p.add_argument('--summaries',nargs='+',required=True)
    model=p.add_mutually_exclusive_group(required=True)
    model.add_argument('--checkpoint',help='checkpoint used by every evaluation shard (identified by its SHA256)')
    model.add_argument('--sha256',help='SHA256 of that checkpoint, instead of --checkpoint')
    p.add_argument('--data-root',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    ready=load_data(a.data_root);expected=dataset_rows(ready);checkpoint_sha=a.sha256 or sha(a.checkpoint)
    shards=[];spec_id=None;selector=None;cursor=0
    clipdir=Path(tempfile.mkdtemp(prefix='clip_cache_',dir=Path(a.out).parent))
    clipframes=collections.Counter()
    paths=sorted(map(Path,a.summaries),key=lambda f:read(f)['index_range'][0])
    for path in paths:
        summary=read(path);invocation=read(path.parent/'invocation.json');artifact=read(path.parent/'artifacts.json')
        if artifact['summary_sha256']!=sha(path):raise ValueError('summary hash differs from completion artifact')
        if invocation['spec']['split']!='test':raise ValueError('tracking requires test split')
        if Path(invocation['data_root']).resolve()!=Path(ready['data_root']).resolve():raise ValueError('evaluation used a different dataset path')
        cp=invocation['checkpoint']
        if cp.get('sha256')!=checkpoint_sha:
            raise ValueError('evaluation checkpoint identity mismatch')
        sid=invocation['spec']['id'];sel=invocation['spec']['selector']
        if spec_id is None:spec_id,selector=sid,sel
        if (sid,sel)!=(spec_id,selector):raise ValueError('mixed detector specifications')
        lo,hi=summary['index_range']
        if lo!=cursor or hi<=lo or hi>len(expected):raise ValueError('overlapping, missing or invalid shard interval')
        if summary['images']!=hi-lo or summary['split_images']!=len(expected):raise ValueError('summary frame count mismatch')
        cache=path.with_suffix('')/'detections.jsonl';count=0
        for frame,exp in zip(rows(cache),expected[lo:hi],strict=True):
            validate_frame(frame,exp);count+=1
            with open(clipdir/(frame['seq']+'.jsonl'),'a') as f:
                f.write(json.dumps(frame,allow_nan=False)+'\n')
            clipframes[frame['seq']]+=1
        if count!=hi-lo:raise ValueError('detection frame count mismatch')
        shards.append(dict(summary=str(path.resolve()),summary_sha256=sha(path),
                           invocation_sha256=sha(path.parent/'invocation.json'),
                           path=str(cache.resolve()),sha256=sha(cache),index_range=[lo,hi],frames=count))
        cursor=hi
    if cursor!=len(expected):raise ValueError('final shard is missing')
    index=[{k:r[k] for k in ('seq','frame_id','file','height','width')} for r in expected]
    result=dict(schema=1,status='complete',data_fingerprint=ready['fingerprint'],
        data_root=ready['data_root'],checkpoint_sha256=checkpoint_sha,
        detector_spec_id=spec_id,selector=selector,frames=cursor,clips=len({r['seq'] for r in expected}),
        frame_shape_fingerprint=digest(index),shards=shards,
        clip_files={name:dict(path=str((clipdir/(name+'.jsonl')).resolve()),
            sha256=sha(clipdir/(name+'.jsonl')),frames=count) for name,count in clipframes.items()})
    dump(a.out,result);print(result)

if __name__=='__main__':main()
