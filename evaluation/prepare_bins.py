#!/usr/bin/env python3
import argparse,json,hashlib
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def shard_summaries(path):
    path=Path(path);doc=json.loads(path.read_text())
    if 'index_range' in doc:return [path]
    if 'input_hashes' not in doc:raise ValueError('not an evaluator summary: '+str(path))
    shards=[]
    for child,expected in doc['input_hashes'].items():
        if sha(child)!=expected:raise ValueError('shard summary changed: '+child)
        shards+=shard_summaries(child)
    return shards


def build(data,split,full,query):
    ann=Path(data)/'annotations'/f'panoptic_{split}.json';d=json.loads(ann.read_text())
    images={x['id']:x for x in d['images']};names={x['file_name']:x for x in d['images']}
    suffixes={tuple(Path(k).parts[-3:]):k for k in names}
    if len(suffixes)!=len(names):raise ValueError('ambiguous image suffix')
    expected={}
    for a in d['annotations']:
        im=images[a['image_id']]
        for seg in a['segments_info']:
            if seg['category_id'] in (1,2) and not seg.get('iscrowd',0):expected[(im['file_name'],seg['id'])]=(seg['category_id']-1,seg['area'])
    result={'clusters':sorted({str(x['original_video_id']) for x in d['images']}),'full':[],'query':[],
        'provenance':{'split':split,'annotation_sha256':sha(ann),'files':{}}}
    checkpoints=None
    for side,inputs in (('full',full),('query',query)):
        for shard in [s for p in inputs for s in shard_summaries(p)]:
            summary=json.loads(shard.read_text());lo,hi=summary['index_range']
            if summary['split']!=split or summary['images']!=hi-lo or summary['split_images']!=len(images):raise ValueError('shard population incomplete')
            current=summary['evaluation_identity']['checkpoints']
            if checkpoints is not None and current!=checkpoints:raise ValueError('checkpoint differs between inputs')
            checkpoints=current
            f=shard.with_suffix('')/'instance_matches.jsonl';result['provenance']['files'][str(f)]=sha(f)
            for line in f.read_text().splitlines():
                r=json.loads(line);parts=Path(r['file_name']).parts
                name=suffixes.get(tuple(parts[-3:]))
                if name is None:raise ValueError('absent image')
                im=names[name];k=(name,r['gt_segment_id'])
                if k not in expected or expected[k]!=(r['class_id'],r['original_area']):raise ValueError('GT population mismatch')
                if (r['image_width'],r['image_height'])!=(im['width'],im['height']):raise ValueError('image shape mismatch')
                result[side].append({**r,'file_name':name,'cluster':str(im['original_video_id'])})
    for side in ['full','query']:
        keys=[(r['file_name'],r['gt_segment_id']) for r in result[side]]
        if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('missing or duplicate GT records')
    result['provenance']['checkpoints']=checkpoints
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Build the binned_ci.py bundle from a full-frame run and a re-observation run.')
    p.add_argument('--data',required=True,help='data_panoptic_track directory')
    p.add_argument('--split',default='test',choices=['val','test'])
    p.add_argument('--full',nargs='+',required=True,help='summary.json of the full-frame run (run_eval shards or a postprocess.py aggregate)')
    p.add_argument('--query',nargs='+',required=True,help='summary.json of the re-observation run (shards or aggregate)')
    p.add_argument('--out',required=True);a=p.parse_args()
    result=build(a.data,a.split,a.full,a.query)
    with Path(a.out).open('x') as f:json.dump(result,f,allow_nan=False)
