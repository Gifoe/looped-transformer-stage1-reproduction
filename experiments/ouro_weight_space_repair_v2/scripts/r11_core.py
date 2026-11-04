"""R11 input/source gates. Every held-out parse follows own frozen locks."""
import json
from r7_core import ROOT,read,sha,digest,jsonlines

EXTRA=('r11_core.py','r11_inference.py','r11_worker.py','r11_evaluate.py','r11_chain.py','r11_supervise.py',
       'r10_core.py','r7_inference.py','r7_math.py','runtime.py','ouro_cached_scoring.py','weight_edits.py')


def save(path,value):
    p=ROOT/path;assert p.name.startswith('r11_') and p.parent.is_dir() and p.resolve().is_relative_to(ROOT.resolve())
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf8');tmp.replace(p)


def extra_sources():return {n:sha('scripts/'+n) for n in EXTRA}


def gate(phase):
    protocol=read('configs/r11_PROTOCOL.json');definition=read('configs/r11_CANDIDATE_DEFINITION_LOCK.json')
    assert {n:sha('scripts/'+n) for n in protocol['source_sha256']}==protocol['source_sha256']
    assert definition['protocol_sha256']==sha('configs/r11_PROTOCOL.json') and definition['inference_definition_sha256']==protocol['source_sha256']
    label={'parity':'CANDIDATE','confirmation':'CANDIDATE','validation':'VALIDATION','profile':'FINAL_METHOD','test':'FINAL_METHOD'}[phase]
    path='configs/r11_'+label+'_LOCK.json';lock=read(path)
    candidate=read('configs/r11_CANDIDATE_LOCK.json')
    assert candidate['protocol_sha256']==sha('configs/r11_PROTOCOL.json') and candidate['definition_lock_sha256']==sha('configs/r11_CANDIDATE_DEFINITION_LOCK.json')
    assert candidate['extra_source_sha256']==extra_sources() and candidate['inference_definition_sha256']==protocol['source_sha256']
    assert lock['protocol_sha256']==sha('configs/r11_PROTOCOL.json') and lock['definition_lock_sha256']==sha('configs/r11_CANDIDATE_DEFINITION_LOCK.json')
    assert lock['extra_source_sha256']==extra_sources() and lock['inference_definition_sha256']==protocol['source_sha256']
    names=[lock['primary']] if phase in ['profile','test'] else lock['candidates']
    assert set(names)<=set(definition['candidates']) and len(names)<=2
    assert candidate['candidates']==definition['candidates'] and set(names)<=set(candidate['candidates'])
    if label!='CANDIDATE':assert lock['candidate_lock_sha256']==sha('configs/r11_CANDIDATE_LOCK.json')
    if phase=='test':
        assert lock['test_access_authorized']
        public=read('logs/r11_PUBLIC_FINAL_LOCK_RECEIPT.json')
        assert public['status']=='verified_public' and public['final_method_lock_sha256']==sha(path) and public['repository']=='Gifoe/looped-transformer-stage1-reproduction' and len(public['commit'])==40
    return protocol,lock,path,names


def rows(phase):
    protocol,lock,lockpath,names=gate(phase)
    split={'parity':'discovery','profile':'discovery','confirmation':'confirmation','validation':'validation','test':'test'}[phase]
    manifest='data/slcd_'+split+'.jsonl'
    path='results/r7_'+split+('_development' if split in ['discovery','selection','confirmation'] else '_final')+'.jsonl'
    assert sha(manifest)==protocol['dataset_sha256'][split] and sha(path)==protocol['r7_score_file_sha256'][split]
    receipt=read(path.replace('.jsonl','_COMPLETE.json'));assert receipt['status']=='complete' and receipt['score_file_sha256']==sha(path)
    data=list(jsonlines(manifest));expected={'discovery':384,'confirmation':517,'validation':299,'test':1172}[split]
    assert len(data)==expected
    refs={r['id']:r for r in jsonlines(path)};assert set(refs)=={r['id'] for r in data}
    if phase in ['parity','profile']:data=data[:(2 if phase=='parity' else 12)]
    result=[]
    from slcd_r6_scores import selected_score
    import numpy as np
    for item in data:
        ref=refs[item['id']];assert ref['signature']==receipt['signature'] and ref['gold']==item['gold'] and ref['prompt_sha256']==item['prompt_sha256'] and ref['choice_char_lengths']==item['choice_char_lengths']
        orig,prior=ref['original_methods'],ref['prior_methods'][0]
        assert np.array_equal(selected_score(orig,prior,'r6_prior_projection'),ref['baseline_scores']['M0_R6'])
        result.append(dict(split=split,item=item,original=orig,prior=prior,baseline_scores=ref['baseline_scores']))
    return result
