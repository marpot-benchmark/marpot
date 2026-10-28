#!/usr/bin/env python3
import argparse,collections,gzip,json,os,sys
from pathlib import Path
import numpy as np
from run_eval import dump,sha


def load_stats(path):
    def load(file,ancestors):
        file=Path(file).resolve()
        if file in ancestors:raise ValueError('cyclic statistical provenance')
        ancestors=ancestors|{file};doc=json.loads(file.read_text());result={};files={}
        if 'native_clip_statistics' in doc:
            for name,row in doc['native_clip_statistics'].items():
                classes=np.asarray(row['classes'],dtype=np.float64);small=np.asarray(row['small'],dtype=np.float64)
                if classes.shape!=(4,4) or small.shape!=(2,2):raise ValueError('invalid native statistic dimensions')
                if np.any(small[:,1]>small[:,0]):raise ValueError('small TP exceeds GT')
                result[str(name)]=np.concatenate([classes[:2],small[:,1:2],small[:,:1]-small[:,1:2],np.zeros((2,1))],axis=1)
                files[str(name)]=list(row['image_files'])
        elif 'clips' in doc:
            for name,row in doc['clips'].items():
                classes=np.asarray(row['classes'],dtype=np.float64);sizes=np.asarray(row['sizes'],dtype=np.float64)
                if classes.shape!=(4,4) or sizes.shape!=(2,3,5):raise ValueError('invalid PanSR native statistic dimensions')
                result[str(name)]=np.concatenate([classes[:2],sizes[:2,0][:,[1,3,2]]],axis=1)
                if 'image_files' not in doc:raise ValueError('native per-clip image coverage is required')
                files[str(name)]=[doc['image_files'][str(i)] for i in row['image_ids']]
        elif 'input_hashes' in doc:
            for child,expected in doc['input_hashes'].items():
                if sha(child)!=expected:raise ValueError('native shard statistics changed')
                rows,images=load(child,ancestors)
                for clip,values in rows.items():
                    result.setdefault(clip,np.zeros((2,7)))[:] += values
                    files.setdefault(clip,[]).extend(images[clip])
        else:raise ValueError('native clip sufficient statistics missing; legacy sequence_rows are not PQ: '+str(file))
        for clip,values in result.items():
            names=files[clip]
            if not names or len(set(names))!=len(names):raise ValueError('missing or duplicate clip frame coverage')
            if any(Path(name).parent.name!=clip for name in names):raise ValueError('noncanonical clip identity')
            if not np.isfinite(values).all() or np.any(values<0):raise ValueError('invalid native sufficient statistics')
        return result,files
    result,_=load(path,set());keys=sorted(result)
    if not keys:raise ValueError('no native clip sufficient statistics')
    return keys,np.stack([result[k] for k in keys])


def metrics(summed):
    summed=np.asarray(summed,dtype=np.float64)
    if summed.shape[-2:]!=(2,7) or not np.isfinite(summed).all():raise ValueError('expected two native thing-class statistics')
    den=summed[...,0]+.5*(summed[...,1]+summed[...,2])
    per_class=np.divide(summed[...,3],den,out=np.zeros_like(den),where=den>0)
    pq=per_class.mean(-1)
    tp=summed[...,4].sum(-1);fn=summed[...,5].sum(-1)
    recall=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=tp+fn>0)
    return np.stack([pq,recall],axis=-1)


def load_mask_iou_recall(path):
    def load(file,ancestors):
        file=Path(file).resolve()
        if file in ancestors:raise ValueError('cyclic legacy recall provenance')
        ancestors=ancestors|{file};doc=json.loads(file.read_text());rows={}
        if 'sequence_rows' in doc:
            for row in doc['sequence_rows']:
                clip=str(row['sequence']);rows.setdefault(clip,np.zeros(2))
                if row['size']=='small' and int(row['class_id']) in (0,1):
                    tp,gt,fn=float(row['tp']),float(row['gt']),float(row['fn'])
                    if min(tp,gt,fn)<0 or tp+fn!=gt:raise ValueError('invalid ordinary mask-IoU recall counts')
                    rows[clip]+=[tp,gt]
        elif 'clips' in doc:
            return None
        elif 'input_hashes' in doc:
            for child,expected in doc['input_hashes'].items():
                if sha(child)!=expected:raise ValueError('legacy recall shard changed')
                child_rows=load(child,ancestors)
                if child_rows is None:return None
                for clip,values in child_rows.items():rows.setdefault(clip,np.zeros(2))[:] += values
        else:return None
        return rows or None
    rows=load(path,set())
    if rows is None:return None
    keys=sorted(rows);return keys,np.stack([rows[k] for k in keys])


def group_original_videos(keys, values, data):
    if not data:return keys,values
    mapping={}
    for split in ('val','test'):
        path=Path(data)/'annotations'/f'panoptic_{split}.json'
        if not path.exists():continue
        for image in json.loads(path.read_text())['images']:
            clip=Path(image['file_name']).parent.name;original=str(image.get('original_video_id',clip))
            if clip in mapping and mapping[clip]!=original:raise ValueError('Original video identity changed within run')
            mapping[clip]=original
    missing=set(keys)-set(mapping)
    if missing:raise ValueError('Bootstrap clips missing from annotation: '+str(sorted(missing)))
    groups={}
    for i,key in enumerate(keys):groups.setdefault(mapping[key],[]).append(i)
    names=sorted(groups)
    return names,np.stack([values[groups[name]].sum(0) for name in names])


def paired(a):
    raw_bases=[load_stats(p) for p in a.baseline];raw_cands=[load_stats(p) for p in a.candidate]
    if any(k!=raw_bases[0][0] for k,_ in raw_bases+raw_cands):raise ValueError('paired clip identities differ')
    data=getattr(a,'data',None)
    bases=[group_original_videos(k,v,data) for k,v in raw_bases]
    cands=[group_original_videos(k,v,data) for k,v in raw_cands]
    keys=bases[0][0]
    if any(k!=keys for k,_ in bases+cands):raise ValueError('paired clip identities differ')
    if not keys:raise ValueError('no clips')
    rng=np.random.default_rng(a.seed);draw=rng.integers(0,len(keys),(a.replicates,len(keys)))
    def score(rows,draw=None):
        return np.mean([metrics(x.sum(0) if draw is None else x[draw].sum(1)) for _,x in rows],axis=0)
    point=100*(score(cands)-score(bases));samples=100*(score(cands,draw)-score(bases,draw))
    contrasts={name:dict(delta=float(point[i]),ci95=np.quantile(samples[:,i],[.025,.975]).tolist())
        for i,name in enumerate(('PQ_Th','small_recall_native_void'))}
    legacy=[load_mask_iou_recall(p) for p in a.baseline+a.candidate]
    legacy_available=all(row is not None for row in legacy)
    if legacy_available:
        legacy=[group_original_videos(k,v,data) for k,v in legacy]
        if any(row[0]!=keys for row in legacy):raise ValueError('legacy/native clip identities differ')
        def legacy_score(rows,draw=None):
            scores=[]
            for _,value in rows:
                counts=value.sum(0) if draw is None else value[draw].sum(1)
                scores.append(np.divide(counts[...,0],counts[...,1],out=np.zeros_like(counts[...,0]),where=counts[...,1]>0))
            return np.mean(scores,axis=0)
        lb,lc=legacy[:len(a.baseline)],legacy[len(a.baseline):]
        delta=100*(legacy_score(lc)-legacy_score(lb));replicates=100*(legacy_score(lc,draw)-legacy_score(lb,draw))
        contrasts['small_recall_mask_iou']=dict(delta=float(delta),ci95=np.quantile(replicates,[.025,.975]).tolist())
    dump(Path(a.out)/'summary.json',dict(cluster_ids=keys,clip_ids=raw_bases[0][0],bootstrap_unit='original_video_id' if data else 'continuity_run',replicates=a.replicates,seed=a.seed,
        unit='percentage_points',contrasts=contrasts,
        metric_definitions=dict(PQ_Th='Native GT-void-aware PQ; fixed mean over ship and obstacle in every draw',
            small_recall_native_void='Native GT-void-aware IoU > 0.5 matches divided by GT thing instances with area < 1024',
            small_recall_mask_iou='Original EoMT/M2F sequence_rows ordinary mask-IoU > 0.5 recall; same definition as legacy table and calibration'),
        legacy_mask_iou_available=legacy_available,
        input_hashes={str(f):sha(f) for f in a.baseline+a.candidate},
        note='Paired source-video bootstrap (legacy clip fallback without data);  predictor/seed averaging occurs after pooled-statistic scoring. '
             +('Both recall definitions are reported separately.' if legacy_available else
               'Ordinary mask-IoU recall is unavailable for at least one input (including PanSR); the native recall CI does not represent legacy table R_S.')))


def dataset(a):
    result={}
    for split in ('train','val','test'):
        f=Path(a.data)/'annotations'/f'panoptic_{split}.json';doc=json.loads(f.read_text())
        imgs={im['id']:im for im in doc['images']};cats=collections.Counter();pixels=collections.Counter()
        sizes=collections.Counter();tracked=collections.Counter();formats=collections.Counter()
        for im in imgs.values():formats[f'{im["width"]}x{im["height"]}']+=1
        for ann in doc['annotations']:
            for s in ann['segments_info']:
                cid=int(s['category_id']);area=int(s['area']);cats[cid]+=1;pixels[cid]+=area
                if s.get('iscrowd',0):continue
                if cid in (1,2):
                    sizes[(cid,'small' if area<1024 else 'medium' if area<9216 else 'large')]+=1
                    if s.get('track_id') is not None and int(s.get('track_id',-1))>=0:tracked[cid]+=1
        result[split]=dict(frames=len(imgs),clips=len({Path(im['file_name']).parent.name for im in imgs.values()}),
            instances_per_category=dict(cats),pixels_per_category=dict(pixels),
            thing_sizes={str(k):v for k,v in sizes.items()},identity_bearing=dict(tracked),formats=dict(formats),annotation_sha256=sha(f))
    dump(Path(a.out)/'summary.json',dict(splits=result,small_definition='native area < 1024 pixels',
        medium_definition='1024 <= area < 9216 pixels',large_definition='area >= 9216 pixels'))


def figures(a):
    values={};inputs={};routes={}
    edges=np.array([1,19,379,7378,143643,np.inf])
    for label,paths in [('full',a.baseline),('query',a.candidate)]:
        bins=np.zeros((2,5,2),np.int64);byshape={};routelist=[]
        for summary in paths:
            f=Path(summary);inputs[str(f)]=sha(f)
            side=f.with_suffix('');matches=side/'instance_matches.jsonl'
            if not matches.exists():raise ValueError('native per-instance audit missing '+str(matches))
            for line in matches.read_text().splitlines():
                r=json.loads(line);cid=r['class_id']
                if cid not in (0,1):continue
                b=int(np.searchsorted(edges,r['original_area'],side='right')-1)
                b=max(0,min(4,b));bins[cid,b,0]+=1;bins[cid,b,1]+=int(r['matched'])
                fmt=f'{r["image_width"]}x{r["image_height"]}'
                x=byshape.setdefault(fmt,np.zeros((2,5,2),np.int64));x[cid,b,0]+=1;x[cid,b,1]+=int(r['matched'])
            route=side/'frame_routes.jsonl'
            for line in route.read_text().splitlines():
                r=json.loads(line);routelist.append({'file_name':r['file_name'],'selected_indices':r['selected_indices'],
                    'candidate_windows':r['candidate_windows']})
        values[label]=dict(bins=bins.tolist(),formats={k:v.tolist() for k,v in byshape.items()})
        routes[label]=routelist
    if not np.array_equal(np.asarray(values['full']['bins'])[:,:,0],np.asarray(values['query']['bins'])[:,:,0]):
        raise ValueError('figure populations differ')
    dump(Path(a.out)/'summary.json',dict(edges=[1,19,379,7378,143643,'inf'],
          counts_gt_tp=values,input_hashes=inputs,note='Raw native-area counts for plotting; corrected-label statistics replace old values.'))
    dump(Path(a.out)/'window_routes.json',routes)


def residual(a):
    sys.path.insert(0,a.root)
    from datasets.sea_panoptic import SeaPanopticDataset
    from tracking import maskutil
    summaries=[json.loads(Path(s).read_text()) for s in a.inputs]
    splits={x['split'] for x in summaries}
    if len(splits)!=1:raise ValueError('residual inputs come from different splits')
    split=splits.pop()
    ds=SeaPanopticDataset(a.data,split,transforms=None,sequence_mode=True,use_gt_track_ids=True)
    files=[];expected=0
    for s,x in zip(a.inputs,summaries):
        f=Path(s);expected+=x['images'];files+=sorted(f.with_suffix('').glob('candidates/*.json.gz'))
    if len(files)!=expected:raise ValueError('raw candidate cache incomplete')
    counts=collections.Counter();rows=[]
    for f in files:
        r=json.load(gzip.open(f,'rt'));context=r['context']
        if context['file_name']!=ds.samples[context['image_index']][0]:raise ValueError('residual frame differs from the split order')
        _,target=ds._build_sample(context['image_index']);trace=r['trace']
        if tuple(target['masks'].shape[-2:])!=(context['image_height'],context['image_width']):raise ValueError('residual frame size differs')
        for j,mask in enumerate(target['masks']):
            cid=int(target['labels'][j]);gt=mask.numpy().astype(bool);area=int(gt.sum())
            if cid not in (0,1) or area>=1024 or bool(target['is_crowd'][j]):continue
            def best(dets):
                v=0.
                for d in dets:
                    if d['class_id']!=cid:continue
                    m=maskutil.decode(d['mask_rle']);it=int((gt&m).sum());un=area+int(m.sum())-it
                    v=max(v,it/un if un else 0.)
                return v
            final=best(trace['final'])
            if final>.5:continue
            stages=dict(raw=0.,score=0.,nms=0.,merged=best(trace['merged']),final=final)
            for window in trace['raw_windows']:
                y0,x0,y1,x1=window['bounds'];crop=gt[y0:y1,x0:x1]
                for q in window['queries']:
                    if q['class_id']!=cid:continue
                    m=maskutil.decode(q['mask_rle_local']);it=int((crop&m).sum());un=area+int(m.sum())-it;iou=it/un if un else 0.
                    stages['raw']=max(stages['raw'],iou)
                    if q['score_keep']:stages['score']=max(stages['score'],iou)
                    if q['nms_keep']:stages['nms']=max(stages['nms'],iou)
            reason='compose_or_visibility'
            for key,label in [('raw','no_selected_window_raw_recovery'),('score','score_filter'),('nms','local_nms'),('merged','cross_scale_merge')]:
                if stages[key]<=.5:reason=label;break
            counts[reason]+=1;rows.append(dict(**r['context'],gt_segment_id=int(target['seg_ids'][j]),class_id=cid,
                area=area,first_blocking_stage=reason,best_same_class_iou_by_stage=stages))
    dump(Path(a.out)/'summary.json',dict(split=split,frames=len(files),counts=dict(counts),failures=rows,input_hashes={str(f):sha(f) for f in a.inputs}))


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['paired','dataset','figures','residual'])
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]),help='repository root (default: parent of evaluation/)')
    p.add_argument('--data');p.add_argument('--out',required=True)
    p.add_argument('--baseline',nargs='+');p.add_argument('--candidate',nargs='+');p.add_argument('--inputs',nargs='+')
    p.add_argument('--replicates',type=int,default=10000);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True);globals()[a.operation](a)


if __name__=='__main__':main()
