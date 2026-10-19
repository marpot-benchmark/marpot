#!/usr/bin/env python3
import argparse
import collections
import concurrent.futures
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tracking_eval.common import dataset_rows,digest,dump,load_data,read,sha,write_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True)
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--verify-reference-frames',type=int,default=4);a=p.parse_args()
    import numpy as np
    from PIL import Image
    from tracking import maskutil
    ready=load_data(a.data_root);data=Path(ready['data_root']);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    manifest=out/'truth_manifest.json'
    if manifest.exists():
        prior=read(manifest)
        if prior.get('truth_protocol_version')!=2:raise ValueError('Legacy truth cache lacks released-class/source-video protocol; choose a new output directory')
        if prior['data_fingerprint']!=ready['fingerprint']:raise ValueError('truth cache belongs to different labels')
        if sha(prior['path'])!=prior['sha256']:raise ValueError('truth cache modified')
        for item in prior['clips'].values():
            if sha(item['path'])!=item['sha256']:raise ValueError('per-clip ground truth modified')
        return
    source=dataset_rows(ready);start=time.perf_counter()
    evaluated_classes=sorted({int(s['category_id'])-1 for r in source for s in r['annotation']['segments_info']
        if s['category_id'] in (1,2) and int(s.get('track_id',-1))>=0 and not s.get('iscrowd',0)})
    if not evaluated_classes:raise ValueError('No valid tracked ground truth classes')
    def frame(row):
        rgb=np.array(Image.open(data/'panoptic'/row['annotation']['file_name']).convert('RGB'),dtype=np.int64)
        ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2]
        if list(ids.shape)!=[row['height'],row['width']]:raise ValueError('truth dimensions differ from frozen image')
        valid=[];ignored=[]
        for seg in row['annotation']['segments_info']:
            if seg['category_id'] not in (1,2):continue
            mask=ids==seg['id']
            if not mask.any():continue
            rle=maskutil.encode(mask);x,y,w,h=maskutil.to_bbox(rle)
            d=dict(class_id=seg['category_id']-1,track_id=int(seg.get('track_id',-1)),
                   bbox=[x,y,x+w,y+h],mask_rle=rle)
            (valid if d['track_id']>=0 and not seg.get('iscrowd',0) else ignored).append(d)
        return {**{k:row[k] for k in ('index','seq','frame_id','file','width','height','original_video_id')},
                'detections':valid,'ignored':ignored}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        truth=list(pool.map(frame,source))
    verified=0
    if a.verify_reference_frames:
        from datasets.sea_panoptic import SeaPanopticDataset
        dataset=SeaPanopticDataset(str(data),'test',transforms=None,sequence_mode=True,use_gt_track_ids=True)
        for index in range(min(a.verify_reference_frames,len(source))):
            _,target=dataset._build_sample(index);original=[]
            for j,mask in enumerate(target['masks']):
                cid=int(target['labels'][j])
                if cid in (0,1):original.append((cid,int(target['pseudo_track_ids'][j]),maskutil.encode(mask.numpy().astype(bool))))
            actual=[(d['class_id'],d['track_id'],d['mask_rle']) for group in ('detections','ignored') for d in truth[index][group]]
            norm=lambda values:sorted(values,key=lambda x:(x[0],x[1],x[2]['counts']))
            if norm(original)!=norm(actual):raise ValueError('mask-only truth parser differs from original dataset')
            verified+=1
    write_rows(out/'truth.jsonl',truth);clips=collections.defaultdict(list)
    for item in truth:clips[item['seq']].append({k:item[k] for k in ('seq','frame_id','detections')})
    clipmeta={}
    for name,frames in clips.items():
        path=out/'gt'/(name+'.jsonl');write_rows(path,sorted(frames,key=lambda f:f['frame_id']))
        clusters={r['original_video_id'] for r in source if r['seq']==name}
        if len(clusters)!=1:raise ValueError('A continuity run spans multiple original videos')
        clipmeta[name]=dict(path=str(path.resolve()),sha256=sha(path),frames=len(frames),original_video_id=clusters.pop())
    index=[{k:r[k] for k in ('seq','frame_id','file','height','width')} for r in source]
    result=dict(schema=1,truth_protocol_version=2,status='complete',data_fingerprint=ready['fingerprint'],
        data_root=str(data),path=str((out/'truth.jsonl').resolve()),sha256=sha(out/'truth.jsonl'),
        frames=len(truth),evaluated_classes=evaluated_classes,truth_type='panoptic_mask_tracking',frame_shape_fingerprint=digest(index),clips=clipmeta,
        original_loader_parity_frames=verified,wall_seconds=time.perf_counter()-start,
        protocol='Panoptic mask truth (not MOT box truth); nonnegative noncrowd IDs; negative/crowd IDs ignored; evaluate only classes with valid test identities')
    dump(manifest,result);print(result)

if __name__=='__main__':main()
