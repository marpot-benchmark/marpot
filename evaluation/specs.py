def eomt_specs():
    rows=[]
    def add(name,split,selector='query-small',window=1120,k=3,**options):
        rows.append(dict(id=f'eomt_{split}_{name}',family='eomt',split=split,
                         selector=selector,window=window,k=k,options=options))
    for split in ('val','test'):
        add('full896',split,'full',tracking=split=='test')
        add('query',split,tracking=split=='test',save_candidates=True,raw_trace=True)
        add('horizon',split,'horizon-row')
        add('center',split,'center')
        for seed in range(3):add(f'random{seed}',split,'random',seed=seed)
        add('all',split,'all')
        add('full_thr03',split,'full',full_threshold=.3)
        if split=='val':add('full_thr05',split,'full',full_threshold=.5)
        for size in (1344,1792):add(f'full{size}',split,'full',infer_size=size)
        add('density',split,density=True)
        add('entropy',split,predictive_entropy=True)
    variants={
        'retained':{'query_set':'retained'},'score_qualified':{'query_set':'score_qualified'},
        'floor001':{'floor':.01},'floor02':{'floor':.2},'floor05':{'floor':.5},
        'gamma0':{'gamma':0},'gamma1':{'gamma':1},'mean':{'pool':'mean'},
        'max':{'pool':'max'},'sum':{'pool':'sum'},'no_clear':{'clear':False},
        'semantic_prob':{'semantic_prob':True}}
    for name,opt in variants.items():add(name,'val',calibration_group='eomt_evidence_variants',**opt)
    add('retained','test',**variants['retained'])
    add('val_min_recall','test',select_from='eomt_evidence_variants',select_rule='min_small_recall')
    add('val_max_recall','test',select_from='eomt_evidence_variants',select_rule='max_small_recall')
    for k in (1,2,4):add(f'budget{k}','val',k=k)
    for window in (560,840,1400):
        add(f'query_w{window}','val',window=window,save_candidates=True,raw_trace=True)
        add(f'all_w{window}','val','all',window=window)
        add(f'query_w{window}','test',window=window)
        add(f'all_w{window}','test','all',window=window)
    add('center_w560','test','center',window=560)
    for seed in range(3):add(f'random{seed}_w560','test','random',window=560,seed=seed)
    assert len(rows)==61
    return rows


def m2f_specs():
    rows=[]
    for split in ('val','test'):
        for name,selector in [('full','full'),('horizon','horizon-row'),('all','all')]:
            rows.append(dict(id=f'm2f_{split}_{name}',family='m2f',split=split,
                selector=selector,window=1120,k=3,options={} if selector=='full' else
                {'select_from':'m2f_calibration','select_rule':'max_pq_then_recall'}))
        if split=='test':rows.append(dict(id='m2f_test_query',family='m2f',split=split,
            selector='query-small',window=1120,k=3,options={'select_from':'m2f_calibration','select_rule':'max_pq_then_recall'}))
    for floor in (.01,.05,.2):
        for score in (.8,.9,.95):rows.append(dict(id=f'm2f_val_f{floor}_s{score}',family='m2f',split='val',
            selector='query-small',window=1120,k=3,options=dict(floor=floor,local_score=score,calibration_group='m2f_calibration')))
    assert len(rows)==16
    return rows


def inventory():
    return eomt_specs()+m2f_specs()


if __name__=='__main__':
    import argparse
    import json
    from pathlib import Path
    p=argparse.ArgumentParser(description='Print the evaluation inventory, or write one spec JSON per id.')
    p.add_argument('--out',help='directory for <spec id>.json files (EoMT and Mask2Former specs, '
                                'the families evaluation/run_eval.py runs)')
    a=p.parse_args()
    if a.out is None:
        print(json.dumps(inventory(),indent=2))
    else:
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        rows=inventory()
        if len({row['id'] for row in rows})!=len(rows):raise ValueError('duplicate spec id')
        for row in rows:(out/(row['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        print(f'wrote {len(rows)} spec files to {out}')
