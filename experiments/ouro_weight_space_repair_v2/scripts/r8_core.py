"""New round, same folder; own data gates, immutable R7 authorities."""
import csv,hashlib,itertools,json
from r7_core import ROOT,sha,read,digest,jsonlines


def save(relative,value):
    p=ROOT/relative;assert p.name.startswith('r8_') and p.parent.is_dir() and p.resolve().is_relative_to(ROOT.resolve())
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)


def csvout(relative,values):
    p=ROOT/relative;assert p.name.startswith('r8_') and p.parent.is_dir()
    with p.open('w',newline='',encoding='utf8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in values for k in row)));w.writeheader();w.writerows(values)


def load_rows(split,need_p5=True):
    protocol=read('configs/r8_PROTOCOL.json')
    if split=='confirmation':assert read('configs/r8_CANDIDATE_LOCK.json')['status']=='frozen_before_current_confirmation'
    elif split=='validation':assert read('configs/r8_VALIDATION_LOCK.json')['status']=='frozen_before_current_validation'
    elif split=='test':assert read('configs/r8_FINAL_METHOD_LOCK.json')['test_access_authorized']
    else:assert split in ['discovery','selection']
    assert sha('data/slcd_'+split+'.jsonl')==protocol['dataset_sha256'][split]
    suffix='_development' if split in ['discovery','selection','confirmation'] else '_final'
    path='results/r7_'+split+suffix+'.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'))
    assert receipt['status']=='complete' and sha(path)==receipt['score_file_sha256']
    assert receipt['score_file_sha256']==protocol['r7_score_file_sha256'][split]
    data=list(jsonlines(path));expected_n={'discovery':384,'selection':192,'confirmation':517,'validation':299,'test':1172}[split]
    assert len(data)==len({r['id'] for r in data})==expected_n
    manifests={r['id']:r for r in jsonlines('data/slcd_'+split+'.jsonl')};assert set(manifests)=={r['id'] for r in data}
    prior={}
    if split in ['discovery','selection','confirmation']:
        p='results/slcd_r5_priors_train.jsonl';pr=read(p.replace('.jsonl','_COMPLETE.json'));assert sha(p)==pr['score_file_sha256']
        with (ROOT/p).open(encoding='utf8') as stream:
            source=stream if split=='confirmation' else itertools.islice(stream,576)
            prior={r['id']:r for r in map(json.loads,source)}
    elif split=='test' and need_p5:
        p='results/slcd_r6_priors_test.jsonl';pr=read(p.replace('.jsonl','_COMPLETE.json'));assert sha(p)==pr['score_file_sha256']
        prior={r['id']:r for r in jsonlines(p)}
    elif split=='validation' and need_p5:
        pr=read('results/r8_validation_prior5_COMPLETE.json');assert pr['status']=='complete'
        assert sha(pr['score_file'])==pr['score_file_sha256'];prior={r['id']:r for r in jsonlines(pr['score_file'])}
    result=[]
    import numpy as np
    from slcd_r6_scores import selected_score
    for row in data:
        assert row['signature']==receipt['signature'];item=manifests[row['id']]
        assert row['prompt_sha256']==item['prompt_sha256'] and row['gold']==item['gold'] and row['choice_char_lengths']==item['choice_char_lengths']
        pm=row['prior_methods'][0]
        if row['id'] in prior:
            pp=prior[row['id']];assert pp['prompt_sha256']==item['prompt_sha256'] and pp['gold']==item['gold'] and pp['choice_char_lengths']==item['choice_char_lengths']
            if 'prior_methods' in pp:pm=pp['prior_methods']
            else:pm={**pm,'native_k5':dict(normalized_scores=(np.asarray(pp['P5_sum_scores'])/np.asarray(item['choice_char_lengths'])).tolist())}
        vec=lambda methods,key:np.asarray(methods[key]['normalized_scores'],dtype=np.float64)
        original=row['original_methods'];r=np.asarray(row['baseline_scores']['M0_R6']);a=np.asarray(row['baseline_scores']['A1'])
        assert np.array_equal(r,selected_score(original,pm,'r6_prior_projection'))
        h=vec(pm,'native_k4')-vec(pm,'native_k5') if need_p5 else None
        result.append(dict(id=row['id'],gold=row['gold'],split=split,item=item,
            a=a,r=r,m2=np.asarray(row['method_scores']['M2_N0_token']),late=vec(original,'native_k4')-vec(original,'native_k5'),prior_late=h,
            baseline_scores=row['baseline_scores'],historical_M1=row['method_scores'].get('M1_multi_candidate'),
            source_score_sha256=receipt['score_file_sha256']))
    return result
