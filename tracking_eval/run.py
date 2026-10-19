#!/usr/bin/env python3
import argparse
import copy
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tracking_eval.common import TRACKERS,dump,frame_key,read,rows,sha,validate_frame,write_rows


def cluster_indices(items):
    groups={}
    for i,item in enumerate(items):groups.setdefault(str(item.get('original_video_id',item['clip'])),[]).append(i)
    keys=sorted(groups)
    return keys,[groups[key] for key in keys]


def filter_tracking_detections(detections,gt,evaluated_classes,decode):
    valid=[];ignored={};kept=[]
    for d in gt['detections']:
        if d['class_id'] not in evaluated_classes:continue
        mask=decode(d['mask_rle']);valid.append((d['class_id'],mask,int(mask.sum())))
    for d in gt['ignored']:
        mask=decode(d['mask_rle']);cid=d['class_id']
        ignored[cid]=mask if cid not in ignored else ignored[cid]|mask
    for d in detections:
        if d['class_id'] not in evaluated_classes:continue
        mask=decode(d['mask_rle']);area=int(mask.sum());priority=False
        for cid,gm,ga in valid:
            if cid!=d['class_id']:continue
            intersection=int((gm&mask).sum());union=ga+area-intersection
            if union and intersection/union>=.05:priority=True;break
        region=ignored.get(d['class_id'])
        discard=not priority and region is not None and area and (region&mask).sum()/area>.5
        if not discard:kept.append(d)
    return kept


def main():
    p=argparse.ArgumentParser();p.add_argument('--truth-manifest',required=True)
    p.add_argument('--full-cache-manifest',required=True);p.add_argument('--query-cache-manifest',required=True)
    p.add_argument('--out',required=True);p.add_argument('--tracker',choices=TRACKERS+['all'],required=True)
    p.add_argument('--replicates',type=int,default=10000);a=p.parse_args()
    import numpy as np
    from tracking import maskutil
    from tracking.build import track_sequence
    from tools.run_track import load_tracker_config,make_frame_loader,_drop_short_tracks
    from tools.eval_track import evaluate
    from tools.hota import official_hota
    truth=read(a.truth_manifest);manifests={'full':read(a.full_cache_manifest),'query':read(a.query_cache_manifest)}
    if truth['status']!='complete' or sha(truth['path'])!=truth['sha256']:raise ValueError('truth cache incomplete or modified')
    for name,manifest in manifests.items():
        for key in ('data_fingerprint','frames','frame_shape_fingerprint'):
            if manifest.get(key)!=truth[key]:raise ValueError(f'{name} cache and truth differ in {key}')
        if manifest['status']!='complete' or set(manifest['clip_files'])!=set(truth['clips']):raise ValueError('cache clips incomplete')
    if manifests['full']['checkpoint_sha256']!=manifests['query']['checkpoint_sha256']:raise ValueError('paired detections must use same checkpoint')
    if manifests['full']['selector']!='full' or manifests['query']['selector']!='query-small':raise ValueError('expected full and query-small detector pair')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    evaluated_classes=truth.get('evaluated_classes',[0,1])
    if not evaluated_classes or any(c not in (0,1) for c in evaluated_classes):raise ValueError('Invalid evaluated tracking classes')
    truth_by_seq={}
    for frame in rows(truth['path']):truth_by_seq.setdefault(frame['seq'],{})[frame_key(frame)]=frame
    trackers=TRACKERS if a.tracker=='all' else [a.tracker];results={};start=time.perf_counter()
    for tracker in trackers:
        cfg_path=ROOT/'configs/tracking'/(tracker+'.yaml');cfg=load_tracker_config(str(cfg_path))
        needed=[]
        if cfg.use_query_embed:needed.append('query_embed')
        if cfg.appearance_enabled:needed.append(cfg.appearance_source)
        for detector,manifest in manifests.items():
            perclip=[]
            for seq,item in sorted(manifest['clip_files'].items()):
                if sha(item['path'])!=item['sha256']:raise ValueError('detection clip cache modified')
                source=sorted(rows(item['path']),key=lambda f:f['frame_id'])
                expected=truth_by_seq[seq]
                if len(source)!=len(expected) or {frame_key(f) for f in source}!=set(expected):raise ValueError('missing or duplicate frames within clip')
                for fr in source:
                    validate_frame(fr,expected[frame_key(fr)])
                    if cfg.sea_prior_enabled and fr.get('seasurface_rle') is None:
                        raise ValueError('sea-prior tracker requires exported seasurface mask')
                    for d in fr['detections']:
                        if any(k not in d for k in needed):raise ValueError(f'{detector}/{tracker} lacks required embedding')
                        if any(not np.isfinite(np.asarray(d[k],dtype=float)).all() or not len(d[k]) for k in needed):
                            raise ValueError('nonfinite or empty tracking embedding')
                tracked=track_sequence(copy.deepcopy(source),cfg,make_frame_loader(str(Path(truth['data_root'])/'images')))
                if cfg.min_track_len>1:tracked=_drop_short_tracks(tracked,cfg.min_track_len)
                if [frame_key(f) for f in tracked]!=[frame_key(f) for f in source]:raise ValueError('tracker changed frame identity/order')
                filtered=[]
                for fr in tracked:
                    kept=filter_tracking_detections(fr['detections'],expected[frame_key(fr)],evaluated_classes,maskutil.decode)
                    filtered.append({**fr,'detections':kept})
                pred=out/detector/tracker/(seq+'.jsonl');write_rows(pred,filtered)
                gtpath=truth['clips'][seq]['path']
                if sha(gtpath)!=truth['clips'][seq]['sha256']:raise ValueError('ground truth clip changed')
                metrics={str(cid):evaluate(gtpath,str(pred),.5,'mask',cid) for cid in evaluated_classes}
                hota={str(cid):official_hota(gtpath,pred,cid) for cid in evaluated_classes}
                perclip.append(dict(clip=seq,original_video_id=truth['clips'][seq].get('original_video_id',seq),metrics=metrics,hota=hota))
            results[detector+'_'+tracker]=perclip
            dump(out/(detector+'_'+tracker+'.json'),dict(clips=perclip,evaluated_classes=evaluated_classes,truth_type='panoptic_mask_tracking',config_sha256=sha(cfg_path),
                protocol='nonnegative valid track IDs; same-class negative-ID IoA>0.5 ignore, valid-GT IoU>=0.05 priority; CLEAR/IDF1 at mask IoU>=0.5; official TrackEval HOTA implementation'))
    rng=np.random.default_rng(0);idf_intervals=[];hota_intervals=[];summary={}
    def sufficient(items):
        return np.array([[sum(m.get(k,0) for m in item['metrics'].values()) for k in ('IDTP','n_gt','n_pred','FP','FN','IDsw')] for item in items],dtype=float)
    def terms(items):
        numer=[];denom=[]
        for item in items:
            h=[item['hota'][str(c)] for c in evaluated_classes]
            numer.append([np.array(x['AssA'])*np.array(x['HOTA_TP']) for x in h])
            denom.append([np.array(x['HOTA_TP'])+np.array(x['HOTA_FP'])+np.array(x['HOTA_FN']) for x in h])
        return np.array(numer),np.array(denom)
    for tracker in trackers:
        base=results['full_'+tracker];cand=results['query_'+tracker]
        if [r['clip'] for r in base]!=[r['clip'] for r in cand]:raise ValueError('bootstrap clip pairing differs')
        clusters,groups=cluster_indices(base)
        pool=lambda values:np.stack([values[ix].sum(0) for ix in groups])
        b=pool(sufficient(base));c=pool(sufficient(cand));draw=rng.integers(0,len(b),(a.replicates,len(b)))
        def score(values):
            x=values[draw].sum(1);return 2*x[:,0]/np.maximum(x[:,1]+x[:,2],1)
        delta=score(c)-score(b)
        idf_intervals.append(dict(tracker=tracker,delta_IDF1_CI95=np.quantile(delta,[.025,.975]).tolist(),replicates=a.replicates))
        bn,bd=map(pool,terms(base));cn,cd=map(pool,terms(cand));diff=[]
        for lo in range(0,a.replicates,1000):
            ix=rng.integers(0,len(b),(min(1000,a.replicates-lo),len(b)))
            bf=np.sqrt(bn[ix].sum(1)/np.maximum(bd[ix].sum(1),1e-10)).mean((1,2))
            cf=np.sqrt(cn[ix].sum(1)/np.maximum(cd[ix].sum(1),1e-10)).mean((1,2));diff.extend((cf-bf).tolist())
        hota_intervals.append(dict(tracker=tracker,delta_HOTA_CI95=np.quantile(diff,[.025,.975]).tolist(),delta_macro_HOTA_CI95=np.quantile(diff,[.025,.975]).tolist(),evaluated_classes=evaluated_classes,replicates=a.replicates))
        for detector,values,nn,dd in [('full',b,bn,bd),('query',c,cn,cd)]:
            totals=values.sum(0)
            summary[detector+'_'+tracker]=dict(IDF1=2*totals[0]/max(totals[1]+totals[2],1) if totals[1] else None,
                MOTA=1-(totals[3]+totals[4]+totals[5])/totals[1] if totals[1] else None,
                HOTA=float(np.sqrt(nn.sum(0)/np.maximum(dd.sum(0),1e-10)).mean()),
                macro_HOTA=float(np.sqrt(nn.sum(0)/np.maximum(dd.sum(0),1e-10)).mean()),
                IDTP=int(totals[0]),n_gt=int(totals[1]),n_pred=int(totals[2]),
                FP=int(totals[3]),FN=int(totals[4]),IDsw=int(totals[5]))
    dump(out/'paired_bootstrap.json',idf_intervals);dump(out/'official_hota_paired.json',hota_intervals)
    dump(out/'summary.json',dict(status='complete',trackers=trackers,results=summary,frames=truth['frames'],
        clips=len(truth['clips']),bootstrap_clusters=len(clusters),evaluated_classes=evaluated_classes,
        class_names=['ship' if c==0 else 'obstacle' for c in evaluated_classes],truth_type='panoptic_mask_tracking',data_fingerprint=truth['data_fingerprint'],
        checkpoint_sha256=manifests['full']['checkpoint_sha256'],wall_seconds=time.perf_counter()-start,
        input_hashes={str(p):sha(p) for p in (a.truth_manifest,a.full_cache_manifest,a.query_cache_manifest)},
        native_helper_hashes={str(p):sha(p) for p in [ROOT/'tools'/f for f in
            ('run_track.py','eval_track.py','hota.py')]},
        bootstrap_unit='original_video_id; continuity runs have independent track states and are sampled together; full/query paired',
        scope='custom MarPOT mask tracking identity-ignore protocol; only GT identity-bearing classes; not native MOT box leaderboard equivalence'))
    print(json.dumps(summary))

if __name__=='__main__':main()
