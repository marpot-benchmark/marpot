import json,hashlib,argparse
from pathlib import Path
import numpy as np

def compute(bundle,replicates=10000,seed=0):
 if replicates<100:raise ValueError('at least 100 replicates required')
 if not bundle['clusters'] or len(set(bundle['clusters']))!=len(bundle['clusters']):raise ValueError('invalid clusters')
 records={}
 for side in ['full','query']:
  rows={}
  for r in bundle[side]:
   k=(r['file_name'],r['gt_segment_id'])
   if k in rows:raise ValueError('duplicate GT identity')
   if r['cluster'] not in bundle['clusters'] or r['class_id'] not in (0,1) or r['original_area']<=0 or type(r['matched']) is not bool:raise ValueError('invalid record')
   rows[k]=r
  records[side]=rows
 if not records['full'] or records['full'].keys()!=records['query'].keys():raise ValueError('paired population differs')
 for k,r in records['full'].items():
  q=records['query'][k]
  if any(r[x]!=q[x] for x in ['cluster','class_id','original_area','image_width','image_height']):raise ValueError('paired GT metadata differs')
 clusters=sorted(bundle['clusters']);ci={c:i for i,c in enumerate(clusters)}
 arrays={}
 for k,r in records['full'].items():
  area=r['original_area'];sz='small' if area<1024 else 'medium' if area<9216 else 'large'
  edge=int(np.searchsorted([1,19,379,7378,143643],area,side='right')-1)
  cls='ship' if r['class_id']==0 else 'obstacle';fmt=f"{r['image_width']}x{r['image_height']}"
  keys=['overall',f'size/{sz}',f'class_size/{cls}/{sz}',f'area5/{edge}',f'class_area5/{cls}/{edge}',f'resolution/{fmt}/{sz}']
  for label in keys:
   x=arrays.setdefault(label,np.zeros((len(clusters),3),dtype=np.int64))
   x[ci[r['cluster']]] += [1,int(r['matched']),int(records['query'][k]['matched'])]
 rng=np.random.default_rng(seed);draws=rng.integers(0,len(clusters),(replicates,len(clusters)))
 result={}
 for label,x in sorted(arrays.items()):
  gt,f,q=x.sum(0);pooled=x[draws].sum(1);valid=pooled[:,0]>0
  delta=100*(pooled[valid,2]-pooled[valid,1])/pooled[valid,0]
  result[label]={'n_gt':int(gt),'clusters_with_gt':int((x[:,0]>0).sum()),'full_tp':int(f),'query_tp':int(q),
   'full_recall_pct':100*f/gt,'query_recall_pct':100*q/gt,'delta_pp':100*(q-f)/gt,
   'ci95_pp':np.quantile(delta,[.025,.975]).tolist() if valid.any() else None,
   'valid_replicates':int(valid.sum()),'zero_denominator_replicates':int((~valid).sum()),
   'sparse_stratum':bool((x[:,0]>0).sum()<2 or valid.mean()<.95)}
 return {'bootstrap_unit':'original_video_id','replicates':replicates,'seed':seed,'clusters':clusters,
  'direction':'query minus full','small_definition':'area < 1024','medium_definition':'1024 <= area < 9216',
  'area5_edges':[1,19,379,7378,143643,'inf'],'units':'percentage points',
  'note':'Fixed checkpoint, paired percentile intervals; no multiplicity correction. Zero-denominator replicates excluded and counted; sparse strata are descriptive, not significance claims.',
  'source':bundle.get('provenance',{}),'results':result}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);p.add_argument('--replicates',type=int,default=10000);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
 src=Path(a.bundle);b=json.loads(src.read_text());r=compute(b,a.replicates,a.seed);r['input_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
 with Path(a.out).open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
