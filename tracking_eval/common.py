import hashlib
import json
import os
import re
from pathlib import Path

TRACKERS=['m1_maskiou_ablation','m1_motion_iou','m2_motion_query',
          'm3_byte_kalman_ocsort','m4a_appearance','m5_reid','m6_byte_sea_gmc']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def read(path):return json.loads(Path(path).read_text())

def rows(path):
    with open(path) as f:
        for line in f:
            if line.strip():yield json.loads(line)

def write_rows(path,items):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'w') as f:
        for item in items:f.write(json.dumps(item,allow_nan=False)+'\n')
    os.replace(tmp,path)

def load_data(root):
    root=Path(root).resolve();annotations=root/'annotations/panoptic_test.json'
    if not annotations.is_file():raise FileNotFoundError(annotations)
    return dict(data_root=str(root),fingerprint=sha(annotations))

def dataset_rows(ready):
    data=Path(ready['data_root'])
    doc=read(data/'annotations/panoptic_test.json')
    ims={im['id']:im for im in doc['images']};result=[];keys=set()
    for ann in doc['annotations']:
        if not any(s['category_id'] in (1,2,3,4) for s in ann['segments_info']):continue
        im=ims[ann['image_id']];path=Path(im['file_name']);m=re.search(r'(\d+)$',path.stem)
        if not m:raise ValueError('unparseable frame number: '+str(path))
        seq,frame=path.parent.name,int(m.group(1));key=(seq,frame)
        if key in keys:raise ValueError('duplicate clip/frame identity')
        keys.add(key)
        result.append(dict(index=len(result),seq=seq,frame_id=frame,file=im['file_name'],
                           width=im['width'],height=im['height'],annotation=ann,
                           original_video_id=str(im.get('original_video_id',seq))))
    return result

def frame_key(frame):return (frame['seq'],int(frame['frame_id']))

def validate_frame(frame,expected):
    for key in ('seq','frame_id','file','width','height'):
        if frame.get(key)!=expected[key]:raise ValueError(f'detection {key} does not match frozen dataset: {frame.get(key)} != {expected[key]}')
    for d in frame['detections']:
        if d['class_id'] not in (0,1):raise ValueError('tracking detections must contain thing classes only')
        if d['mask_rle']['size']!=[frame['height'],frame['width']]:raise ValueError('mask dimensions differ')
