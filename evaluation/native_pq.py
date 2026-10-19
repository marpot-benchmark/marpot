#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

NAMES=['ship','obstacle','seasurface','backdrop']

def size_bin(area):return 0 if area<1024 else 1 if area<9216 else 2

def normalize(ids,info,offset):
    ids=np.asarray(ids,dtype=np.int64)
    unique,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
    meta={int(x['id']):x for x in info}
    mapping=np.zeros(len(unique),dtype=np.int32);segments={};stuff={}
    nextid=1
    for j,(sid,area) in enumerate(zip(unique,counts)):
        if int(sid) not in meta:
            if sid!=0:raise ValueError('unlisted nonvoid segment '+str(sid))
            continue
        raw=meta[int(sid)];c=int(raw['category_id'])-offset
        if not 0<=c<4:raise ValueError('invalid category')
        crowd=bool(raw.get('iscrowd',0))
        if offset==0 and crowd:raise ValueError('predictions cannot declare crowd')
        if c>=2 and not crowd and c in stuff:newid=stuff[c]
        else:
            newid=nextid;nextid+=1
            segments[newid]={'class_id':c,'area':0,'iscrowd':crowd}
            if c>=2 and not crowd:stuff[c]=newid
        mapping[j]=newid;segments[newid]['area']+=int(area)
    return mapping[inv].reshape(ids.shape),segments

def evaluate_native(gt_ids,gt_info,pred_ids,pred_info):
    if gt_ids.shape!=pred_ids.shape:raise ValueError('native shape mismatch')
    gt,g=normalize(gt_ids,gt_info,1);pred,p=normalize(pred_ids,pred_info,0)
    base=len(p)+1
    pairs,count=np.unique(gt.astype(np.int64)*base+pred,return_counts=True)
    intersections={ (int(k//base),int(k%base)):int(n) for k,n in zip(pairs,count)}
    classes=np.zeros((4,4),np.float64);sizes=np.zeros((2,3,5),np.float64)
    gm=set();pm=set()
    for (gi,pi),inter in intersections.items():
        if gi==0 or pi==0 or g[gi]['iscrowd'] or g[gi]['class_id']!=p[pi]['class_id']:continue
        union=g[gi]['area']+p[pi]['area']-inter-intersections.get((0,pi),0)
        iou=inter/union if union else 0.
        if iou>.5:
            assert gi not in gm and pi not in pm
            gm.add(gi);pm.add(pi);c=g[gi]['class_id'];classes[c,0]+=1;classes[c,3]+=iou
            if c<2:
                b=size_bin(g[gi]['area']);sizes[c,b,1]+=1;sizes[c,b,4]+=iou
    for gi,v in g.items():
        if v['iscrowd']:continue
        c=v['class_id'];miss=gi not in gm
        if miss:classes[c,2]+=1
        if c<2:
            b=size_bin(v['area']);sizes[c,b,0]+=1;sizes[c,b,3]+=int(miss)
    for pi,v in p.items():
        if pi in pm:continue
        ignored=intersections.get((0,pi),0)+sum(intersections.get((gi,pi),0) for gi,gt in g.items()
            if gt['iscrowd'] and gt['class_id']==v['class_id'])
        if ignored/v['area']>.5:continue
        c=v['class_id'];classes[c,1]+=1
        if c<2:sizes[c,size_bin(v['area']),2]+=1
    return classes,sizes

def metrics(classes,sizes):
    a=np.asarray(classes);s=np.asarray(sizes);den=a[:,0]+.5*(a[:,1]+a[:,2]);valid=den>0
    pq=np.divide(a[:,3],den,out=np.zeros(4),where=valid)
    sq=np.divide(a[:,3],a[:,0],out=np.zeros(4),where=a[:,0]>0)
    rq=np.divide(a[:,0],den,out=np.zeros(4),where=valid)
    out={}
    for key,x in [('PQ',pq),('SQ',sq),('RQ',rq)]:
        for suffix,idx in [('',np.arange(4)),('_Th',np.arange(2)),('_St',np.arange(2,4))]:
            ii=idx[valid[idx]];out[key+suffix]=float(x[ii].mean()) if len(ii) else 0.
        for c,name in enumerate(NAMES):out[key+'_'+name]=float(x[c])
    for b,name in enumerate(['S','M','L']):out['R_'+name]=float(s[:,b,1].sum()/s[:,b,0].sum()) if s[:,b,0].sum() else 0.
    out.update(Small_GT=int(s[:,0,0].sum()),Small_TP=int(s[:,0,1].sum()),Thing_TP=int(a[:2,0].sum()),Thing_FP=int(a[:2,1].sum()),Thing_FN=int(a[:2,2].sum()))
    return out


def read_ids(path):
    from PIL import Image
    a=np.asarray(Image.open(path),dtype=np.int64)
    return a[...,0]+256*a[...,1]+65536*a[...,2] if a.ndim==3 else a


def frame_key(file_name,split):
    parts=Path(file_name).with_suffix('').parts
    if len(parts)>2 and parts[0]==split:parts=parts[1:]
    return '/'.join(parts)


def load_predictions(pred_json,pred_dir,split,category_base):
    pred_dir=Path(pred_dir);by_id={};by_name={}
    if pred_json is None:
        for meta in sorted(pred_dir.glob('*.json')):
            doc=json.loads(meta.read_text())
            if 'source_file' not in doc or 'segments_info' not in doc:continue
            key=frame_key(doc['source_file'],split)
            if key in by_name:raise ValueError('duplicate prediction for '+key)
            by_name[key]=(meta.parent/doc['file_name'],[dict(s) for s in doc['segments_info']])
        return by_id,by_name
    for entry in json.loads(Path(pred_json).read_text())['annotations']:
        segments=[{**s,'category_id':int(s['category_id'])-category_base} for s in entry['segments_info']]
        item=(pred_dir/entry['file_name'],segments)
        if 'image_id' in entry:
            if entry['image_id'] in by_id:raise ValueError('duplicate prediction for image_id '+str(entry['image_id']))
            by_id[entry['image_id']]=item
        else:
            key=frame_key(entry['file_name'],split)
            if key in by_name:raise ValueError('duplicate prediction for '+key)
            by_name[key]=item
    return by_id,by_name


def evaluate_split(data,split,pred_json,pred_dir,category_base=1):
    doc=json.loads((Path(data)/'annotations'/f'panoptic_{split}.json').read_text())
    anns={a['image_id']:a for a in doc['annotations']}
    by_id,by_name=load_predictions(pred_json,pred_dir,split,category_base)
    classes=np.zeros((4,4));sizes=np.zeros((2,3,5));clips={};image_ids=[];image_files={};used=set()
    for image in doc['images']:
        gt=anns[image['id']];key=frame_key(image['file_name'],split)
        if image['id'] in by_id:png,segments=by_id[image['id']];used.add(('id',image['id']))
        elif key in by_name:png,segments=by_name[key];used.add(('name',key))
        else:raise ValueError('missing prediction for '+image['file_name'])
        c,s=evaluate_native(read_ids(Path(data)/'panoptic'/gt['file_name']),gt['segments_info'],read_ids(png),segments)
        classes+=c;sizes+=s
        clip=clips.setdefault(Path(image['file_name']).parent.name,dict(classes=np.zeros((4,4)),sizes=np.zeros((2,3,5)),image_ids=[]))
        clip['classes']+=c;clip['sizes']+=s;clip['image_ids'].append(image['id'])
        image_ids.append(image['id']);image_files[str(image['id'])]=image['file_name']
    unused=len(by_id)+len(by_name)-len(used)
    if unused:raise ValueError(f'{unused} predictions do not match any {split} image')
    return dict(split=split,images=len(image_ids),num_clips=len(clips),units='fractions (0..1); counts integer',
        metrics=metrics(classes,sizes),classes=classes.tolist(),sizes=sizes.tolist(),
        image_ids=image_ids,image_files=image_files,
        clips={k:dict(classes=v['classes'].tolist(),sizes=v['sizes'].tolist(),image_ids=v['image_ids']) for k,v in clips.items()},
        protocol=dict(resolution='native image size',match='same class, IoU > 0.5; prediction pixels on GT void are removed from the union',
            ignore='GT void (id 0) and crowd segments; an unmatched prediction is not a FP if > 50% of it lies on void or same-class crowd',
            stuff='same-class stuff segments merged; PQ_Th/PQ_St/PQ are means over classes with a nonzero denominator',
            sizes='native GT area: S < 1024 <= M < 9216 <= L pixels (R_S, R_M, R_L)'))


def main():
    p=argparse.ArgumentParser(description='Native-resolution PQ / PQ_Th / PQ_St and R_S / R_M / R_L of panoptic predictions on MarPOT.')
    p.add_argument('--data',required=True,help='data_panoptic_track directory (annotations/, panoptic/)')
    p.add_argument('--split',default='test',choices=['train','val','test'])
    p.add_argument('--pred-json',help='COCO panoptic prediction JSON; omit to read an evaluation/run_eval.py --export directory')
    p.add_argument('--pred-dir',required=True,help='directory that the prediction file_name entries are relative to')
    p.add_argument('--category-base',type=int,choices=[0,1],default=1,
                   help='1: category ids 1 ship, 2 obstacle, 3 seasurface, 4 backdrop (as in the annotations); 0: ids 0-3')
    p.add_argument('--out',help='write the full result (metrics, per-class and per-clip statistics) to this JSON file')
    a=p.parse_args()
    result=evaluate_split(a.data,a.split,a.pred_json,a.pred_dir,a.category_base)
    m=result['metrics']
    for key in ('PQ','PQ_Th','PQ_St','SQ','RQ','R_S','R_M','R_L'):print(f'{key:6s} {100*m[key]:7.2f}')
    print(f'Thing TP {m["Thing_TP"]}  FP {m["Thing_FP"]}  FN {m["Thing_FN"]}  (small GT {m["Small_GT"]}, small TP {m["Small_TP"]})')
    print(f'{result["images"]} images, {result["num_clips"]} clips')
    if a.out:
        out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
