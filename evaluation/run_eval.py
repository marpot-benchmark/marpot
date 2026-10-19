#!/usr/bin/env python3
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time
import types
if __package__:
    from .evaluation_identity import build_identity,fingerprint
else:
    from evaluation_identity import build_identity,fingerprint


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def checkpoint_record(path,expected_sha256=None):
    ckpt=Path(path);digest=sha(ckpt)
    if expected_sha256 and digest!=expected_sha256.lower():raise RuntimeError('checkpoint hash mismatch: '+str(ckpt))
    return ckpt,{'path':str(ckpt),'sha256':digest}


def resolve_options(spec,selection):
    spec=json.loads(json.dumps(spec));opts=spec['options']
    if opts.get('select_from'):
        if not selection:raise ValueError('this test rule requires --selection (validation selection JSON written by postprocess.py select)')
        s=json.loads(Path(selection).read_text())
        key=opts['select_from']+':'+opts['select_rule']
        if key not in s['selections']:raise ValueError('missing selection '+key)
        entry=s['selections'][key]
        if entry['split']!='val':raise ValueError('selection must use validation only')
        spec['options']={**entry['options'],**{k:opts[k] for k in ('tracking','save_candidates','raw_trace','trace') if k in opts}}
        spec['selection_provenance']=entry
    return spec


class NativeMetricObserver:
    def __init__(self,dataset,start,end):
        self.dataset=dataset;self.index=start;self.start=start;self.end=end
        self.frames=[];self.clips={};self.metric=None

    @staticmethod
    def snapshot(metric):
        import numpy as np
        names=('true_positives','false_positives','false_negatives','iou_sum')
        states={n:getattr(metric,n).detach().cpu().numpy().copy() for n in names}
        return np.asarray([[states[n][metric.cat_id_to_continuous_id[c]] for n in names] for c in range(4)],dtype=np.float64)

    def install(self,model,metric):
        self.metric=metric;original=model.update_metrics_panoptic
        namespace=inspect.unwrap(original.__func__).__globals__
        if '_get_color_areas' not in namespace or '_calculate_iou' not in namespace:
            raise RuntimeError('Unrecognized native metric implementation; cannot infer PQ from diagnostic rows')
        def observed(_model,preds,targets,is_crowds,block_idx):
            import numpy as np
            if len(preds)!=1 or len(targets)!=1 or not self.index<self.end:
                raise RuntimeError('Native metric observation expects one frame per update')
            before=self.snapshot(metric);areas=[];matched=set()
            old_areas=namespace['_get_color_areas'];old_iou=namespace['_calculate_iou']
            def area_observer(*args,**kwargs):
                value=old_areas(*args,**kwargs);areas.append(value);return value
            def iou_observer(pred_color,target_color,*args,**kwargs):
                value=old_iou(pred_color,target_color,*args,**kwargs)
                if float(value)>.5:matched.add(tuple(target_color))
                return value
            namespace['_get_color_areas']=area_observer;namespace['_calculate_iou']=iou_observer
            try:result=original(preds,targets,is_crowds,block_idx)
            finally:namespace['_get_color_areas']=old_areas;namespace['_calculate_iou']=old_iou
            if len(areas)!=3:raise RuntimeError('Native metric area-call contract changed')
            small=np.zeros((2,2),np.int64)
            for color,area in areas[1].items():
                cid=int(color[0]);instance=int(color[1])
                if cid not in (0,1):continue
                if bool(is_crowds[0][instance]):continue
                if int(area)<1024:
                    small[cid,0]+=1;small[cid,1]+=int(tuple(color) in matched)
            delta=self.snapshot(metric)-before
            if np.any(delta[:,:3]<0) or np.any(delta[:,3]<-1e-8):raise RuntimeError('Native metric state reset within a shard')
            file=str(self.dataset.samples[self.index][0]);clip=Path(file).parent.name
            row=dict(index=self.index,file_name=file,clip=clip,classes=delta.tolist(),small=small.tolist())
            self.frames.append(row)
            total=self.clips.setdefault(clip,dict(classes=np.zeros((4,4)),small=np.zeros((2,2),np.int64),image_files=[]))
            total['classes']+=delta;total['small']+=small;total['image_files'].append(file)
            self.index+=1
            return result
        model.update_metrics_panoptic=types.MethodType(observed,model)

    def finish(self):
        import numpy as np
        if self.index!=self.end or len(self.frames)!=self.end-self.start:
            raise RuntimeError('Incomplete native metric observation')
        combined=sum((np.asarray(r['classes']) for r in self.frames),np.zeros((4,4)))
        if not np.allclose(combined,self.snapshot(self.metric),rtol=0,atol=1e-6):
            raise RuntimeError('Native per-frame metric deltas do not reconstruct final metric')
        return {k:dict(classes=v['classes'].tolist(),small=v['small'].tolist(),image_files=v['image_files']) for k,v in self.clips.items()}


def main():
    p=argparse.ArgumentParser(description='Evaluate one spec (configs/specs/*.json) on one split or one contiguous shard [start, end).')
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]),help='repository root (default: parent of evaluation/)')
    p.add_argument('--data-root',required=True,help='data_panoptic_track directory (annotations/, images/, panoptic/)')
    p.add_argument('--config',required=True,help='EoMT config, e.g. configs/eomt/eomt_large_896_joint_scratch_cp.yaml')
    p.add_argument('--checkpoint',required=True,help='EoMT checkpoint (also required for Mask2Former specs)')
    p.add_argument('--sha256',help='optional expected SHA-256 of --checkpoint')
    p.add_argument('--spec',required=True)
    p.add_argument('--selection',help='validation selection JSON (postprocess.py select) for specs with select_from')
    p.add_argument('--m2f-config',help='Mask2Former config, e.g. configs/m2f/mask2former_swin_s_marpot.py')
    p.add_argument('--m2f-checkpoint',help='trained Mask2Former checkpoint')
    p.add_argument('--out',required=True);p.add_argument('--start',type=int,default=0)
    p.add_argument('--end',type=int);p.add_argument('--export',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();out=Path(a.out).resolve()
    out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root))
    partial=[out/name for name in ('summary','summary.json','panoptic','artifacts.json','native_frame_statistics.json') if (out/name).exists()]
    if partial:
        raise RuntimeError('output already contains evaluation artifacts; choose a new --out directory')
    os.environ.setdefault('USE_TF','0');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    spec=resolve_options(json.loads(Path(a.spec).read_text()),a.selection)
    ckpt,cp=checkpoint_record(a.checkpoint,a.sha256)
    from datasets.sea_panoptic import SeaPanopticDataset
    ds=SeaPanopticDataset(a.data_root,spec['split'],transforms=None,sequence_mode=True,use_gt_track_ids=True)
    end=min(a.end if a.end is not None else len(ds.samples),len(ds.samples))
    if not 0<=a.start<end:raise ValueError('empty or invalid shard')
    need_trace=bool(a.export or spec['options'].get('tracking') or spec['options'].get('save_candidates') or spec['options'].get('raw_trace'))
    options={'strict_budget':True,'trace':need_trace,'warmup':True,**spec['options']}
    if spec['family']=='m2f':
        if not a.m2f_config or not a.m2f_checkpoint:raise ValueError('Mask2Former specs require --m2f-config and --m2f-checkpoint')
        mckpt,mrecord=checkpoint_record(a.m2f_checkpoint)
        mrecord['config_sha256']=sha(a.m2f_config)
        import m2f.mmdet_marpot
        os.environ['MARPOT_M2F_CONFIG']=str(Path(a.m2f_config).resolve())
        os.environ['MARPOT_M2F_CHECKPOINT']=str(mckpt.resolve())
        options['backend']='m2f';cp['m2f']=mrecord
    if options.get('density') or options.get('predictive_entropy'):
        try:
            from tools.selector_adapter import install
        except ModuleNotFoundError as error:
            if error.name!='tools.selector_adapter':raise
            raise RuntimeError('this spec needs tools/selector_adapter.py (predictive-entropy/density window '
                               'selector), which is not present in this checkout') from None
        install()
    import tools.analyze_panoptic_errors as evaluator
    import tools.reobs_core as core
    observer=NativeMetricObserver(ds,a.start,end)
    original_init_pq=evaluator.init_pq_metric
    def init_pq(model,*args,**kwargs):
        metric=original_init_pq(model,*args,**kwargs);observer.install(model,metric);return metric
    evaluator.init_pq_metric=init_pq
    exports=[]
    if a.export:
        import numpy as np
        from PIL import Image
        state={};old_predict=evaluator.predict;old_record=core.record_frame
        def predict(*args,**kwargs):
            result=old_predict(*args,**kwargs);state['pixels']=result[0].detach().cpu().numpy();return result
        def record(directory,index,context,*args):
            old_record(directory,index,context,*args)
            pixels=state.pop('pixels');h,w=pixels.shape[:2]
            ids=np.zeros((h,w),np.uint32);segments=[]
            for cat,sid in np.unique(pixels.reshape(-1,2),axis=0):
                cat=int(cat);sid=int(sid)
                if not 0<=cat<4 or sid<0:continue
                mask=(pixels[:,:,0]==cat)&(pixels[:,:,1]==sid)
                new_id=len(segments)+1;ids[mask]=new_id;ys,xs=np.where(mask)
                segments.append(dict(id=new_id,category_id=cat,area=int(mask.sum()),
                    bbox=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],iscrowd=0))
            folder=out/'panoptic';folder.mkdir(exist_ok=True)
            name=f'{index:06d}';png=folder/(name+'.png');meta=folder/(name+'.json')
            Image.fromarray(np.stack([ids%256,(ids//256)%256,(ids//65536)%256],-1).astype(np.uint8)).save(png)
            dump(meta,dict(file_name=png.name,source_file=context['file_name'],height=h,width=w,
                category_indexing='zero_based_0_ship_1_obstacle_2_sea_3_backdrop',segments_info=segments))
            exports.extend([dict(path=str(f),sha256=sha(f)) for f in (png,meta)])
        evaluator.predict=predict;core.record_frame=record
    protocol=out/'protocol.json';dump(protocol,options)
    argv=['analyze_panoptic_errors.py','--config',a.config,'--ckpt',str(ckpt),
          '--data',a.data_root,'--split',spec['split'],'--device','cuda:0','--metrics_device','cuda:0',
          '--inference_mode','full-only' if spec['selector']=='full' else 'full+tile',
          '--tile_size',str(spec['window']),str(spec['window']),'--tile_overlap','.2',
          '--tile_score_threshold',str(options.get('local_score',.95)),
          '--tile_merge_threshold','.6','--tile_merge_metric','iomin','--tile_gate_stage','after',
          '--group_by_sequence','--record_tile_details','--record_instance_matches',
          '--supplement-spec',str(protocol),'--index_start',str(a.start),'--index_end',str(end),
          '--out',str(out/'summary.json')]
    if spec['selector'] not in ('all','full'):
        argv+=['--tile_selector',spec['selector'],'--tile_budget',str(spec['k'])]
    if spec['selector']=='query-small':argv+=['--tile_router_resolution','native','--tile_router_min_query_score',str(options.get('floor',.05))]
    if spec['selector']=='random':argv+=['--tile_random_seed',str(options['seed'])]
    if 'infer_size' in options:argv+=['--infer_size',str(options['infer_size'])]
    evaluation_identity=build_identity(root,a.data_root,a.config,cp,spec,options,ds.samples,a.m2f_config)
    dump(out/'invocation.json',dict(argv=argv,spec=spec,checkpoint=cp,split_images=len(ds.samples),
            data_root=a.data_root,source_sha256=sha(__file__),evaluation_identity=evaluation_identity,
            evaluation_fingerprint=fingerprint(evaluation_identity)))
    started=time.time();sys.argv=argv;evaluator.main()
    result=json.loads((out/'summary.json').read_text())
    if result['images']!=end-a.start or result['split_images']!=len(ds.samples):raise RuntimeError('evaluated frame count mismatch')
    if a.export and len(exports)!=2*(end-a.start):raise RuntimeError('missing prediction exports')
    result['evaluation_identity']=evaluation_identity
    result['evaluation_fingerprint']=fingerprint(evaluation_identity)
    result['native_clip_statistics']=observer.finish()
    small_gt=sum(sum(row[0] for row in clip['small']) for clip in result['native_clip_statistics'].values())
    small_tp=sum(sum(row[1] for row in clip['small']) for clip in result['native_clip_statistics'].values())
    result['native_metrics']=dict(PQ_Th=result['pq']['things']['pq'],R_S=small_tp/small_gt if small_gt else 0.,
        Small_GT=small_gt,Small_TP=small_tp,units='fraction',
        definition='Native GT-void-aware matches; fixed two-thing-class PQ; small GT area < 1024 pixels')
    result['native_statistics_protocol']=dict(pq='actual model PanopticQuality state deltas',
        small_recall='GT thing area < 1024 pixels; matches use the native GT-void-aware IoU > 0.5 decisions',
        legacy_small_recall='Original rows and sequence_rows remain unchanged as ordinary mask-IoU diagnostics',
        no_second_image_scoring=True)
    dump(out/'native_frame_statistics.json',dict(protocol=result['native_statistics_protocol'],frames=observer.frames))
    dump(out/'summary.json',result)
    exports.append(dict(path=str(out/'native_frame_statistics.json'),sha256=sha(out/'native_frame_statistics.json')))
    dump(out/'artifacts.json',dict(files=exports,summary_sha256=sha(out/'summary.json'),
           wall_seconds=time.time()-started,native_instance_ap=False,
           note='Panoptic export is not a replacement for native overlapping instance AP.'))


if __name__=='__main__':main()
