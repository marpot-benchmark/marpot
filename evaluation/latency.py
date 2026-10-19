#!/usr/bin/env python3
import argparse, ast, contextlib, copy, csv, datetime, functools, hashlib
import inspect, json, os, random, statistics, sys, textwrap
import time, traceback, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import resolve_options, checkpoint_record

ORDER_REFERENCE = ['full896','full1344','full1792','query','all',
           'query_w560','query_w840','query_w1400','all_w560','all_w840','all_w1400',
           'density','entropy','horizon','m2f_full','m2f_query']
CONFIGS = [name for name in ORDER_REFERENCE if name != 'density']
CLIPS = ['10_s02_8cd213ec','9_s03_c58131bf','11_camera3_s02_cb1281dc','9_s08_86a547dc']
SEED = 20261019
WARMUP = 20

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp')
    with t.open('w') as f:
        json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def read(p): return json.loads(Path(p).read_text())
def status(out,phase,**kw):
    x={'phase':phase,'utc':utc(),'pid':os.getpid(),**kw};dump(out/'STATUS.json',x)
    print(json.dumps(x,ensure_ascii=False),flush=True)

def spec_file(name):
    return ('m2f_test_'+name[4:]) if name.startswith('m2f_') else ('eomt_test_'+name)

def init_runtime(names):
    import torch, numpy as np
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if (ROOT/'tools/selector_adapter.py').is_file():
        from tools.selector_adapter import install
        install()
    elif 'entropy' in names:
        raise RuntimeError('the entropy configuration needs tools/selector_adapter.py, which is not present in this checkout')
    import tools.analyze_panoptic_errors as evaluator
    return torch,np,evaluator

def case_argv(spec,options,args,protocol):
    argv=['analyze_panoptic_errors.py','--config',args.config,'--ckpt',str(args.checkpoint),
          '--data',args.data_root,'--split',spec['split'],'--device','cuda:0','--metrics_device','cuda:0',
          '--inference_mode','full-only' if spec['selector']=='full' else 'full+tile',
          '--tile_size',str(spec['window']),str(spec['window']),'--tile_overlap','.2',
          '--tile_score_threshold',str(options.get('local_score',.95)),
          '--tile_merge_threshold','.6','--tile_merge_metric','iomin','--tile_gate_stage','after',
          '--group_by_sequence','--record_tile_details','--record_instance_matches',
          '--supplement-spec',str(protocol),'--out',str(protocol.with_name('summary.json'))]
    if spec['selector'] not in ('all','full'):
        argv+=['--tile_selector',spec['selector'],'--tile_budget',str(spec['k'])]
    if spec['selector']=='query-small':argv+=['--tile_router_resolution','native','--tile_router_min_query_score',str(options.get('floor',.05))]
    if spec['selector']=='random':argv+=['--tile_random_seed',str(options['seed'])]
    if 'infer_size' in options:argv+=['--infer_size',str(options['infer_size'])]
    return argv

def make_cases(evaluator,names,args,out):
    cases={}
    for name in names:
        path=Path(args.specs_dir)/(spec_file(name)+'.json')
        spec=resolve_options(read(path),args.selection)
        assert spec['split']=='test'
        opts={'strict_budget':True,'trace':False,'warmup':True,**spec['options']}
        if spec['family']=='m2f':opts['backend']='m2f'
        for flag in ['tracking','save_candidates','raw_trace','trace']:opts[flag]=False
        opts['warmup']=False
        protocol=out/'cases'/name/'protocol.json';dump(protocol,opts)
        argv=case_argv(spec,opts,args,protocol)
        a=evaluator.build_parser().parse_args(argv[1:])
        assert a.split=='test' and not a.tile_ckpt and not a.tile_config
        cases[name]={'name':name,'args':a,'options':opts,'spec':spec,'spec_path':str(path),
                     'family':'m2f' if name.startswith('m2f_') else 'eomt',
                     'model_key':('m2f' if name.startswith('m2f_') else str(a.infer_size or 896))}
    dump(out/'resolved_cases.json',{k:{**{x:y for x,y in v.items() if x!='args'},'args':vars(v['args'])} for k,v in cases.items()})
    return cases

def make_pool(data,out):
    import torchvision.tv_tensors as tv
    from PIL import Image,ImageFile
    from datasets.sea_panoptic import SeaPanopticDataset
    ImageFile.LOAD_TRUNCATED_IMAGES=True
    data=Path(data)
    ds=SeaPanopticDataset(data,'test',transforms=None,sequence_mode=True,use_gt_track_ids=True)
    assert len(ds.samples)==2675
    pool=[];images=[]
    for clip in CLIPS:
        for frame in range(1,33):
            filename=f'test/{clip}/{frame:06d}.jpg'
            matches=[i for i,s in enumerate(ds.samples) if s[0]==filename]
            assert len(matches)==1,(filename,matches)
            p=ds.img_root/filename
            with Image.open(p) as im:
                assert im.size==(2800,1600)
                img=tv.Image(im.convert('RGB'))
            pool.append({'pool_index':len(pool),'dataset_index':matches[0],'file_name':filename,
                         'clip':clip,'height':1600,'width':2800,'sha256':sha(p)})
            images.append(img)
    annotation=data/'annotations/panoptic_test.json'
    dump(out/'frame_pool.json',{'split':'test','frames':pool,'annotation_sha256':sha(annotation),
        'seed':SEED,'selection':'Predeclared four clips, frames 1..32 inclusive; no prediction-based selection',
        'warmup_pool_indices':[i*32+j for i in range(4) for j in range(5)]})
    return pool,images

def build_models(cases,args,out):
    import torch
    from tools.eomt_infer import build_inference_model
    ckpt,cp=checkpoint_record(args.checkpoint,args.sha256)
    frozen=torch.load(cp['path'],map_location='cpu',weights_only=True)
    original_load=torch.load; cache_hits=0
    def cached_load(path,*a,**kw):
        nonlocal cache_hits
        if isinstance(path,(str,Path)) and Path(path)==Path(cp['path']):
            cache_hits+=1;return frozen
        return original_load(path,*a,**kw)
    torch.load=cached_load
    models={}; defaults={}
    try:
        for key in ['896','1344','1792','m2f']:
            name={'896':'full896','1344':'full1344','1792':'full1792','m2f':'m2f_full'}[key]
            users=[c for c in cases.values() if c['model_key']==key]
            if not users:continue
            case=cases.get(name,users[0])
            a=case['args']; status(out,'loading_model',model=key)
            m=build_inference_model(a.config,str(cp['path']),device='cuda:0',infer_size=a.infer_size,
                return_query_embed=False,return_mask_pooled=False,reid_enabled=False,
                require_complete_network=True,require_reid_head=False)
            if key=='m2f':
                if not args.m2f_config or not args.m2f_checkpoint:raise ValueError('Mask2Former configurations require --m2f-config and --m2f-checkpoint')
                mckpt,mc=checkpoint_record(args.m2f_checkpoint);mc['config_sha256']=sha(args.m2f_config)
                import m2f.mmdet_marpot
                from tools.m2f_backend import install_backend
                os.environ['MARPOT_M2F_CONFIG']=str(Path(args.m2f_config).resolve())
                os.environ['MARPOT_M2F_CHECKPOINT']=str(mckpt.resolve())
                install_backend(m,case['options'],'cuda:0')
                dump(out/'m2f_checkpoint.json',mc)
            models[key]=m;defaults[key]=(m.mask_thresh,m.overlap_thresh)
    finally:torch.load=original_load
    dump(out/'model_loading.json',{'original_checkpoint_disk_reads':1,'cached_initializations':cache_hits,
      'resolution_instances':[k for k in ['896','1344','1792'] if k in models],'m2f_detector_initializations':int('m2f' in models),
      'checkpoint':cp,'position_embeddings':'Each resolution independently derived by original loader from original checkpoint, never chained resizing',
      'torch_load_restored':torch.load is original_load})
    del frozen
    return models,defaults

def activate(case,models,defaults):
    m=models[case['model_key']];m.supplement_options=copy.deepcopy(case['options'])
    if hasattr(m,'supplement_trace'):delattr(m,'supplement_trace')
    m.mask_thresh,m.overlap_thresh=defaults[case['model_key']]
    if 'full_threshold' in case['options']:m.mask_thresh=float(case['options']['full_threshold'])
    return m

def kwargs_for(a,evaluator):
    thresholds=list(a.tile_score_threshold)
    if len(thresholds)==1:thresholds*=len(evaluator.THING_CLASSES)
    protect=a.tile_protect_full_above_area
    if protect is not None:
        if len(protect)==1:protect=protect*len(evaluator.THING_CLASSES)
        protect={c:v for c,v in zip(evaluator.THING_CLASSES,protect) if v>0} or None
    kw={'tile_size':a.tile_size,'tile_overlap':a.tile_overlap,'score_thresholds':dict(zip(evaluator.THING_CLASSES,thresholds)),
        'mask_threshold':a.tile_mask_threshold,'local_nms_iou':a.tile_local_nms_iou,
        'merge_iou':a.tile_merge_threshold,'merge_mode':a.tile_merge_mode,'merge_metric':a.tile_merge_metric,
        'min_visible_fraction':a.tile_min_visible_fraction,'ownership_max_tile_area':a.tile_ownership_max_tile_area,
        'ownership_min_tile_score':a.tile_ownership_min_tile_score,'ownership_min_full_tile_area_ratio':a.tile_ownership_min_full_tile_area_ratio,
        'tile_max_area':a.tile_max_area,'sea_min_water':a.tile_sea_min_water,'sea_class_id':a.tile_sea_class_id,
        'sea_margin_frac':a.tile_sea_margin_frac,'gate_stage':a.tile_gate_stage,'protect_full_above_area':protect,
        'stats':{},'timing_stats':{}}
    if a.tile_selector.startswith('query-'):
        kw.update(tile_router=a.tile_selector,tile_budget=a.tile_budget,router_top_fraction=a.tile_router_top_fraction,
                  router_min_query_score=a.tile_router_min_query_score,router_resolution=a.tile_router_resolution,selected_indices_sink=[])
        if a.tile_router_min_tile_score is not None:kw['router_min_tile_score']=a.tile_router_min_tile_score
    return kw

def measure(case,model,image,frame,evaluator):
    import torch
    a=case['args'];kw=kwargs_for(a,evaluator)
    torch.cuda.synchronize();start=time.perf_counter()
    if a.tile_selector in ['horizon-row','center','random']:
        kw['tile_indices']=evaluator.select_static_tiles(image.shape[-2],image.shape[-1],a.tile_size,a.tile_overlap,
            a.tile_budget,a.tile_selector,seed=a.tile_random_seed,image_index=frame['dataset_index'])
    legacy_start=time.perf_counter()
    pred,shape=evaluator.predict(model,image,'cuda:0',a.inference_mode,kw,tile_model=None)
    torch.cuda.synchronize();end=time.perf_counter()
    assert tuple(shape)==(1600,2800)
    selected=kw.get('selected_indices_sink',kw.get('tile_indices',None))
    return pred,{'seconds':end-start,'legacy_predict_only_s':end-legacy_start,
      'static_selection_and_call_setup_s':legacy_start-start,
      'selected_indices':list(selected) if selected is not None else None,
      'selected_windows':kw['stats'].get('selected_tiles',0),
      'available_windows':kw['stats'].get('available_tiles',0)}

class StageTimers:
    def __init__(self,model):self.model=model;self.undo=[];self.sums={};self.active=False
    def reset(self):self.sums={};self.active=True
    @contextlib.contextmanager
    def section(self,name):
        import torch
        if not self.active:yield;return
        torch.cuda.synchronize();start=time.perf_counter()
        try:yield
        finally:
            torch.cuda.synchronize();self.sums[name]=self.sums.get(name,0.)+time.perf_counter()-start
    def wrap(self,obj,name,bucket):
        old=getattr(obj,name)
        @functools.wraps(old)
        def call(*a,**kw):
            with self.section(bucket):return old(*a,**kw)
        self.undo.append((obj,name,old));setattr(obj,name,call)
    def split_forward(self,obj,first_bucket,rest_bucket):
        old=obj.forward;tree=ast.parse(textwrap.dedent(inspect.getsource(old)))
        fn=tree.body[0];fn.decorator_list=[]
        assert isinstance(fn.body[0],ast.Assign),ast.dump(fn.body[0])
        def wrap(stmts,bucket):
            return ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='_latency_stages',ctx=ast.Load()),attr='section',ctx=ast.Load()),args=[ast.Constant(bucket)],keywords=[]),optional_vars=None)],body=stmts)
        fn.body=[wrap(fn.body[:1],first_bucket)]+([wrap(fn.body[1:],rest_bucket)] if rest_bucket else fn.body[1:])
        ast.fix_missing_locations(tree);ns={**old.__func__.__globals__,'_latency_stages':self};exec(compile(tree,'<latency_stage_only>','exec'),ns)
        self.undo.append((obj,'forward',old));obj.forward=types.MethodType(ns[fn.name],obj)
    def __enter__(self):
        import tools.eomt_infer as e, tools.tiled_panoptic_infer as t
        import tools.reobs_core as core, tools.analyze_panoptic_errors as ev
        self.wrap(self.model,'resize_and_pad_imgs_instance_panoptic','pre')
        self.split_forward(self.model,'pre',None)
        self.split_forward(self.model.network,'pre','fwd')
        self.wrap(e,'query_routing_evidence','sel');self.wrap(core,'select_supplement_tiles','sel')
        for n in ['merge_same_class_instances','gate_tile_detections','compose_panoptic']:self.wrap(t,n,'comp')
        self.wrap(self.model,'revert_resize_and_pad_logits_instance_panoptic','detail_revert_masks')
        self.wrap(self.model,'to_per_pixel_preds_panoptic','detail_panoptic_decode')
        self.wrap(ev,'panoptic_ids_to_pixels','detail_pixel_conversion')
        return self
    def __exit__(self,*exc):
        self.active=False
        for obj,name,old in reversed(self.undo):setattr(obj,name,old)

def stage_run(cases,models,defaults,pool,images,evaluator,out,probe=False):
    import numpy as np
    rows=[];raw=[];gates=[]
    for name in ['full896','full1344','query']:
        case=cases[name];m=activate(case,models,defaults)
        pred,_=measure(case,m,images[0],pool[0],evaluator)
        expected=hashlib.sha256(pred.numpy().tobytes()).hexdigest();del pred
        with StageTimers(m) as st:
            st.reset();pred,_=measure(case,m,images[0],pool[0],evaluator)
            actual=hashlib.sha256(pred.numpy().tobytes()).hexdigest();del pred
            gates.append({'config':name,'frame':pool[0]['file_name'],'unmodified_pixel_sha256':expected,
                          'instrumented_pixel_sha256':actual,'match':expected==actual})
            dump(out/'stage_equivalence.json',{'checks':gates,'all_match':all(x['match'] for x in gates)})
            if expected!=actual:raise RuntimeError('Stage instrumentation changed prediction: '+name)
            if probe:continue
            warm_indices=[i*32+j for i in range(4) for j in range(5)]
            for i in warm_indices:
                st.reset();pred,_=measure(case,m,images[i],pool[i],evaluator);del pred
            for i in range(128):
                st.reset();pred,rec=measure(case,m,images[i],pool[i],evaluator);del pred
                vals={k:st.sums.get(k,0.) for k in ['pre','fwd','sel','comp']}
                vals['total']=rec['seconds'];vals['other']=vals['total']-sum(vals[k] for k in ['pre','fwd','sel','comp'])
                if vals['other'] < -1e-6:raise RuntimeError('Stage intervals double counted')
                raw.append({'config':name,'pool_index':i,**{k+'_ms':v*1000 for k,v in vals.items()},
                            'detail_ms':{k:v*1000 for k,v in st.sums.items() if k.startswith('detail_')}})
                if i%8==0:status(out,'stage_timing',config=name,frame=i+1,total_frames=128)
        if not probe:
            selected=[r for r in raw if r['config']==name]
            rows.append({'config':name,**{k+'_ms':float(np.mean([r[k+'_ms'] for r in selected])) for k in ['pre','fwd','sel','comp','other','total']}})
            dump(out/'stage_raw.json',raw)
    if not probe:write_csv(out/'stage_breakdown.csv',rows,['config','pre_ms','fwd_ms','sel_ms','comp_ms','other_ms','total_ms'])
    return rows

def write_csv(p,rows,columns):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def summarize(raw,out,names):
    import numpy as np
    rows=[];warnings=[];means={}
    for name in names:
        blocks=[b for b in raw['blocks'] if b['config']==name]
        if len(blocks)!=3:continue
        assert all(len(b['measurements'])==128 for b in blocks)
        repeated=[statistics.mean(x['seconds'] for x in b['measurements']) for b in blocks]
        times=[x['seconds'] for b in blocks for x in b['measurements']]
        means[name]=repeated;mean=statistics.mean(times);spread=(max(repeated)-min(repeated))/mean
        rows.append(dict(config=name,frames=128,repeats=3,mean_s=mean,std_between_repeats_s=statistics.stdev(repeated),
                         p50_s=float(np.quantile(times,.5)),p95_s=float(np.quantile(times,.95))))
        if spread>.05:warnings.append({'config':name,'reason':'repeat_mean_range_over_mean_exceeds_5pct','relative_range':spread,'repeat_means_s':repeated})
    write_csv(out/'latency_summary.csv',rows,['config','frames','repeats','mean_s','std_between_repeats_s','p50_s','p95_s'])
    ratios={}
    for a,b in [('query','full896'),('all','full896'),('all','query'),('query','full1344')]:
        if a in means and b in means:
            ratios[a+'/'+b]={'ratio_of_overall_means':statistics.mean(means[a])/statistics.mean(means[b]),
                             'per_repeat':[x/y for x,y in zip(means[a],means[b])]}
    dump(out/'ratios.json',ratios)
    if 'full896' in means and 'full1344' in means and statistics.mean(means['full896'])>statistics.mean(means['full1344']):
        warnings.append({'reason':'full896_slower_than_full1344','ratio':statistics.mean(means['full896'])/statistics.mean(means['full1344'])})
    dump(out/'anomalies.json',{'alerts':warnings,'definition':'Repeat variation = (max mean-min mean)/overall mean; samples never discarded or replaced'})
    return rows,warnings

def capture_environment(torch,out,pool):
    env={'utc':utc(),'python':sys.version,'torch':torch.__version__,
      'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'cudnn_benchmark':torch.backends.cudnn.benchmark,
      'cudnn_deterministic':torch.backends.cudnn.deterministic,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
      'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),
      'precision':{'eomt':'AMP float16 network forwards; original float32 postprocessing preserved','m2f':'original adapter explicitly disables autocast in detector forward (FP32)'},
      'gpu_visible_count':torch.cuda.device_count(),'gpu':torch.cuda.get_device_name(0),
      'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
      'warmup_frames_per_config_per_repeat':WARMUP,
      'seed':SEED,'frames':128,'repeats':3,'frame_pool':pool,
      'batch_size':1,'model_loading':'Each resolution instance initialized once from the same cached original checkpoint; all instances reused across all randomized rounds',
      'timing_scope':'CPU input tensor -> prediction, including host/device transfer, resize, normalization, forwards, window selection, native postprocessing and composition; no decode, GT scoring or serialization',
      'horizon_boundary_change':'Main time includes static window selection; per-frame legacy_predict_only_s retained for old-boundary comparison',
      'no_training':True}
    dump(out/'environment.json',env)

def main():
    ap=argparse.ArgumentParser(description='Runtime protocol of the paper: batch 1, 128 test frames, 20 warm-up frames, 3 rounds.')
    ap.add_argument('--data-root',required=True,help='data_panoptic_track directory (MarPOT v6.0.1)')
    ap.add_argument('--config',required=True,help='EoMT config, e.g. configs/eomt/eomt_large_896_joint_scratch_cp.yaml')
    ap.add_argument('--checkpoint',required=True,help='EoMT checkpoint');ap.add_argument('--sha256',help='optional expected SHA-256 of --checkpoint')
    ap.add_argument('--specs-dir',default=str(ROOT/'configs/specs'),help='directory written by evaluation/specs.py --out')
    ap.add_argument('--selection',help='validation selection JSON (postprocess.py select); needed for m2f_query')
    ap.add_argument('--m2f-config');ap.add_argument('--m2f-checkpoint')
    ap.add_argument('--configs',nargs='+',choices=CONFIGS,help='subset to time (default: all EoMT configurations; add m2f_full m2f_query explicitly)')
    ap.add_argument('--stages',action='store_true',help='also run the per-stage breakdown of full896, full1344 and query (Table S6)')
    ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    names=a.configs or [n for n in CONFIGS if not n.startswith('m2f_')]
    if a.stages:names=list(dict.fromkeys(names+['full896','full1344','query']))
    names=[n for n in CONFIGS if n in names]
    started=time.time()
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('the runtime benchmark needs a CUDA GPU')
        status(out,'initializing',configs=names)
        torch,np,evaluator=init_runtime(names);cases=make_cases(evaluator,names,a,out)
        pool,images=make_pool(a.data_root,out)
        models,defaults=build_models(cases,a,out)
        capture_environment(torch,out,pool)
        smoke=[]
        for name in names:
            status(out,'all_config_smoke',config=name)
            m=activate(cases[name],models,defaults);pred,rec=measure(cases[name],m,images[0],pool[0],evaluator)
            smoke.append({'config':name,**rec,'pixel_sha256':hashlib.sha256(pred.numpy().tobytes()).hexdigest()});del pred
        dump(out/'all_config_smoke.json',smoke)
        if all(n in names for n in ('full896','full1344','query')):
            stage_run(cases,models,defaults,pool,images,evaluator,out,probe=True)
        orders=[]
        for repeat in range(3):
            order=list(ORDER_REFERENCE);random.Random(SEED+repeat).shuffle(order);orders.append([n for n in order if n in names])
        raw={'schema':'marpot-latency-retest-v1','seed':SEED,'orders':orders,'frames':pool,'blocks':[]}
        dump(out/'latency_raw.json',raw)
        with (out/'latency_incremental.jsonl').open('x') as inc:
            for repeat,order in enumerate(orders):
                for position,name in enumerate(order):
                    case=cases[name];m=activate(case,models,defaults)
                    status(out,'warmup',repeat=repeat+1,config=name,position=position+1,configs_per_repeat=len(order))
                    for i in [i*32+j for i in range(4) for j in range(5)]:
                        pred,_=measure(case,m,images[i],pool[i],evaluator);del pred
                    block={'repeat':repeat+1,'config':name,'position':position+1,'measurements':[]}
                    for i in range(128):
                        pred,rec=measure(case,m,images[i],pool[i],evaluator);del pred
                        row={'pool_index':i,'dataset_index':pool[i]['dataset_index'],'file_name':pool[i]['file_name'],**rec}
                        block['measurements'].append(row);inc.write(json.dumps({'repeat':repeat+1,'config':name,**row})+'\n')
                        if (i+1)%8==0:
                            inc.flush();status(out,'timing',repeat=repeat+1,config=name,position=position+1,frame=i+1,frames=128,
                              completed_blocks=len(raw['blocks']),total_blocks=3*len(order),elapsed_seconds=time.time()-started)
                    raw['blocks'].append(block);dump(out/'latency_raw.json',raw)
                    summarize(raw,out,names)
        rows,warnings=summarize(raw,out,names)
        if a.stages:stage_run(cases,models,defaults,pool,images,evaluator,out)
        status(out,'complete',elapsed_seconds=time.time()-started)
        for r in rows:print(f"{r['config']:12s} mean {1000*r['mean_s']:8.1f} ms  p95 {1000*r['p95_s']:8.1f} ms")
    except BaseException as e:
        dump(out/'FAILED.json',{'status':'FAILED','utc':utc(),'error':repr(e),'traceback':traceback.format_exc(),'partial_results_preserved':True})
        status(out,'failed',error=repr(e));raise

if __name__=='__main__':main()
