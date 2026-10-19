from pathlib import Path
import gzip
import json
import math
import itertools
import numpy as np
import torch

SCHEMA = 'marpot-supplement-v1'

def select_supplement_tiles(payload, windows, budget, fraction, minimum, diagnostics, options):
    from tools.tiled_panoptic_infer import project_windows_to_evidence_grid
    if budget <= 0 or not 0 < fraction <= 1:
        raise ValueError('invalid budget/top fraction')
    residual = payload['evidence'].float().clone()
    if not torch.isfinite(residual).all():
        raise ValueError('routing evidence is not finite')
    projected = project_windows_to_evidence_grid(windows, payload['image_size'], payload['scaled_size'], payload['padded_size'], residual.shape)
    selected, chosen_scores, rounds = [], [], []
    reason = 'budget'
    for _ in range(min(budget, len(windows))):
        scores = []
        for i, (y0,x0,y1,x1) in enumerate(projected):
            v=residual[y0:y1,x0:x1].flatten()
            pool=options.get('pool','top_mean')
            if pool=='top_mean': score=v.topk(max(1,round(v.numel()*fraction))).values.mean()
            elif pool=='mean': score=v.mean()
            elif pool=='sum': score=v.sum()
            elif pool=='max': score=v.max()
            else: raise ValueError('unknown pool '+pool)
            scores.append(float(score))
        eligible=[i for i in range(len(windows)) if i not in selected]
        i=max(eligible,key=lambda j:(scores[j],-j))
        rounds.append({'scores':scores,'eligible':eligible,'best_index':i})
        if minimum is not None and scores[i]<minimum:
            reason='adaptive-threshold';break
        if not options.get('strict_budget',False) and scores[i]<=0:
            reason='no-positive-evidence';break
        selected.append(i);chosen_scores.append(scores[i])
        if options.get('clear',True):
            y0,x0,y1,x1=projected[i];residual[y0:y1,x0:x1]=0
    diagnostics.clear()
    diagnostics.update(stop_reason=reason,selected_tiles=len(selected),candidate_slots=min(budget,len(windows)),rounds=rounds,projected_windows=projected)
    return selected,chosen_scores


def json_dump(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    tmp.replace(path)


def append_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')


def record_frame(directory,index,context,img,target,gt_instances,pred_instances,trace,selected,windows,latency):
    from tools.analyze_panoptic_errors import pair_iou
    from tracking import maskutil
    directory=Path(directory)
    pairs=[];best=[0.0]*len(gt_instances)
    for g,gt in enumerate(gt_instances):
        for p,pr in enumerate(pred_instances):
            if gt['class_id']!=pr['class_id']:continue
            iou=pair_iou(gt['mask'],pr['mask']);best[g]=max(best[g],iou)
            if iou>0.5:pairs.append((iou,g,p))
    matches={};used=set()
    for iou,g,p in sorted(pairs,reverse=True):
        if g not in matches and p not in used:matches[g]=(p,iou);used.add(p)
    final=trace.get('final') or trace.get('full') or []
    for g,gt in enumerate(gt_instances):
        ys,xs=torch.where(gt['mask']);y,x=int(ys[0]),int(xs[0])
        source_index=next((j for j,m in enumerate(target['masks']) if bool(m[y,x])),None)
        segment_id=int(target['seg_ids'][source_index]) if 'seg_ids' in target and source_index is not None else None
        record={**context,'gt_segment_id':segment_id,'class_id':gt['class_id'],'original_area':gt['area'],'matched':g in matches,'best_same_class_iou':best[g],'matched_iou':matches[g][1] if g in matches else None,'sources':[]}
        if g in matches:
            pm=pred_instances[matches[g][0]]['mask'];py,px=pm.nonzero()[0].tolist()
            for det in final:
                if int(det['class_id'])==gt['class_id'] and bool(maskutil.decode(det['mask_rle'])[py,px]):
                    record['sources']=det.get('lineage',[{'source':det.get('source','full'),'query_index':det.get('query_index')}]);break
        append_json(directory/'instance_matches.jsonl',record)
    append_json(directory/'frame_routes.jsonl',{**context,'index':index,'candidate_windows':[list(w) for w in windows],'selected_indices':list(selected),'router':trace.get('router',{}),'full_network_input':trace.get('full_network_input'),'inference_seconds':latency})
    if trace.get('save_candidates'):
        file=directory/'candidates'/f'{index:06d}.json.gz';file.parent.mkdir(parents=True,exist_ok=True)
        tmp=file.with_suffix('.tmp')
        with gzip.open(tmp,'wt') as f:json.dump({'context':context,'trace':trace},f,allow_nan=False)
        tmp.replace(file)
    window_hits={}
    for window in (trace.get('windows',[]) if trace.get('save_candidates') else []):
        hits=[]
        for g,gt in enumerate(gt_instances):
            if gt['area']>=1024:continue
            for d in window['candidates']:
                if d['class_id']!=gt['class_id']:continue
                m=torch.as_tensor(maskutil.decode(d['mask_rle']),device=gt['mask'].device,dtype=torch.bool)
                if pair_iou(gt['mask'],m)>0.5:hits.append(g);break
        window_hits[str(window['window_id'])]=hits
    if window_hits:
        keys=list(window_hits);k=min(3,len(keys))
        combinations=itertools.combinations(keys,k)
        winner=max(combinations,key=lambda c:len(set().union(*(set(window_hits[x]) for x in c))))
        append_json(directory/'window_recovery.jsonl',{**context,'window_gt_indices':window_hits,'oracle_type':'precompose-detection-union','oracle_windows':winner,'oracle_small_hits':len(set().union(*(set(window_hits[x]) for x in winner)))})
    if trace.get('tracking'):
        detections=[]
        for d in final:
            item=dict(d);rle=item['mask_rle'];x,y,w,h=maskutil.to_bbox(rle)
            item['bbox']=[x,y,x+w,y+h];item['score']=float(item.get('tracking_score',item['score']))
            item['centroid']=[x+w/2,y+h/2];item['aspect_ratio']=w/max(h,1e-3)
            if item['score']>=0.05:detections.append(item)
        append_json(directory/'detections.jsonl',{'seq':context['sequence'],'frame_id':context['frame'],'file':context['file_name'],'width':img.shape[-1],'height':img.shape[-2],'detections':detections,'seasurface_rle':trace.get('seasurface_rle'),'protocol':SCHEMA})
