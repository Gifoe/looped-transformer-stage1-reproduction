"""One bounded approximation audit; only existing development inputs are opened."""
import json
from r7_core import ROOT,read,sha,digest,jsonlines

LOCK='configs/r16_NATIVE_KL_PILOT_LOCK.json'


def save(path,value):
    p=ROOT/path;assert p.name.startswith('r16_') and p.parent.is_dir() and p.resolve().is_relative_to(ROOT.resolve())
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)


def gate():
    spec=read(LOCK);assert spec['status']=='frozen_bounded_development_pilot' and spec['new_method_candidates']==1 and not spec['heldout_access_authorized']
    for p,h in spec['source_sha256'].items():assert sha('scripts/'+p)==h,('Source changed',p)
    for p,h in spec['input_sha256'].items():assert sha(p)==h,('Input changed',p)
    assert len(spec['cohort'])==len({r['id'] for r in spec['cohort']})==64
    receipt=read('logs/r16_PUBLIC_DEFINITION_RECEIPT.json');assert receipt['readback_verified'] and receipt['protocol_sha256']==sha(LOCK) and len(receipt['commit'])==40
    return spec


def inputs(spec):
    authority={};manifests={}
    for split in ['discovery','selection']:
        for r in jsonlines('results/r7_'+split+'_development.jsonl'):authority[r['id']]=r
        for r in jsonlines('data/slcd_'+split+'.jsonl'):manifests[r['id']]=r
    states={r['id']:r for r in read('results/slcd_SOURCE_AUDIT.json')['states']};answer=[]
    for chosen in spec['cohort']:
        qid=chosen['id'];old=authority[qid];item=manifests[qid];state=states[qid]
        assert state['split']==chosen['split'] and sha(state['file'])==state['sha256']
        assert item['prompt_sha256']==old['prompt_sha256'] and max(old['baseline_parity_errors'].values())==0
        clean={k:item[k] for k in ['id','request_pairs','choice_char_lengths']}
        expected={k:old[k] for k in ['original_methods','prior_methods','token_diagnostics']}
        expected['M2_scores']=old['method_scores']['M2_N0_token']
        answer.append((chosen,clean,expected,state))
    return answer
