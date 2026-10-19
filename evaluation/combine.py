import collections
import json

from tools.reobs_core import json_dump,SCHEMA


def combine(paths,dest,job):
    parts=[json.loads(p.read_text()) for p in paths]
    n=sum(x['images'] for x in parts)
    sequence_images=collections.Counter(); grouped=collections.defaultdict(lambda:collections.Counter())
    for a in parts:
        sequence_images.update(a['sequence_images'])
        for r in a['sequence_rows']:
            key=(r['sequence'],r['class_id'],r['class_name'],r['size'])
            c=grouped[key]
            for field in ['gt','tp','fp','fn','fn_no_overlap','fn_poor_mask','fn_wrong_class']:c[field]+=r.get(field,0)
            c['iou_sum']+=r['mean_matched_iou']*r['tp']
            c['best_sum']+=r.get('fn_mean_best_same_class_iou',0)*r['fn']
    rows=[]
    for (seq,cid,name,size),c in sorted(grouped.items()):
        rows.append(dict(sequence=seq,class_id=cid,class_name=name,size=size,**{k:c[k] for k in ['gt','tp','fp','fn','fn_no_overlap','fn_poor_mask','fn_wrong_class']},recall=c['tp']/c['gt'] if c['gt'] else 0,mean_matched_iou=c['iou_sum']/c['tp'] if c['tp'] else 0,fn_mean_best_same_class_iou=c['best_sum']/c['fn'] if c['fn'] else 0))
    pc=[]
    for cid in range(4):
        tt=collections.Counter()
        for a in parts:
            r=a['pq']['per_class'][cid]
            for k in ['tp','fp','fn']:tt[k]+=r[k]
            tt['iou']+=r['pq']*(r['tp']+.5*(r['fp']+r['fn']))
        denom=tt['tp']+.5*(tt['fp']+tt['fn'])
        pc.append(dict(class_id=cid,class_name=['ship','obstacle','seasurface','backdrop'][cid],tp=tt['tp'],fp=tt['fp'],fn=tt['fn'],iou_sum=tt['iou'],pq=tt['iou']/denom if denom else 0))
    small_gt=sum(r['gt'] for r in rows if r['size']=='small');small_tp=sum(r['tp'] for r in rows if r['size']=='small')
    selected=sum((a.get('tile_gate_stats') or {}).get('selected_tiles',0) for a in parts)
    if job['selector'] not in ['all','full'] and selected!=n*3:raise RuntimeError(f'budget mismatch {selected} vs {n*3}')
    summary=dict(schema_version=SCHEMA,job=job,images=n,sequence_images=dict(sequence_images),sequence_rows=rows,pq={'per_class':pc,'all':{'pq':sum(r['pq'] for r in pc)/4},'things':{'pq':sum(r['pq'] for r in pc[:2])/2},'stuff':{'pq':sum(r['pq'] for r in pc[2:])/2}},small_gt=small_gt,small_tp=small_tp,small_recall=small_tp/small_gt if small_gt else None,selected_tiles=selected,mean_windows=selected/n,latency={'mean_s':sum(a['latency']['mean_s']*a['images'] for a in parts)/n,'note':'diagnostic run; trace overhead included; see separate latency jobs'},shards=[str(p) for p in paths])
    for size in ['medium','large']:
        den=sum(r['gt'] for r in rows if r['size']==size);num=sum(r['tp'] for r in rows if r['size']==size);summary[size+'_recall']=num/den if den else None
    json_dump(dest,summary);return summary
