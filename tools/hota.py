import json
from pathlib import Path

import numpy as np


def official_hota(gtpath,predpath,cid):
    from trackeval.metrics import HOTA
    from tracking import maskutil
    gt={(r['seq'],r['frame_id']):[d for d in r['detections'] if d['class_id']==cid] for r in map(json.loads,Path(gtpath).read_text().splitlines())}
    pr={(r['seq'],r['frame_id']):[d for d in r['detections'] if d['class_id']==cid] for r in map(json.loads,Path(predpath).read_text().splitlines())}
    gi={x:i for i,x in enumerate(sorted({d['track_id'] for ds in gt.values() for d in ds}))}
    pi={x:i for i,x in enumerate(sorted({d['track_id'] for ds in pr.values() for d in ds}))}
    data={'num_gt_ids':len(gi),'num_tracker_ids':len(pi),'num_gt_dets':sum(map(len,gt.values())),'num_tracker_dets':sum(map(len,pr.values())),'gt_ids':[],'tracker_ids':[],'similarity_scores':[]}
    for key in sorted(set(gt)|set(pr)):
        g=gt.get(key,[]);p=pr.get(key,[])
        gids=np.array([gi[d['track_id']] for d in g],dtype=int);pids=np.array([pi[d['track_id']] for d in p],dtype=int)
        if len(set(gids))!=len(gids) or len(set(pids))!=len(pids):raise RuntimeError('duplicate identities within frame')
        data['gt_ids'].append(gids);data['tracker_ids'].append(pids)
        mat=maskutil.iou_matrix([d['mask_rle'] for d in g],[d['mask_rle'] for d in p]) if g and p else np.zeros((len(g),len(p)))
        data['similarity_scores'].append(mat)
    result=HOTA().eval_sequence(data)
    return {k:v.tolist() if isinstance(v,np.ndarray) else float(v) for k,v in result.items()}
