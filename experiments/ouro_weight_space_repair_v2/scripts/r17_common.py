"""Only12 original development questions; no label/outcome analysis."""
import json
from r7_core import ROOT,read,sha,digest,jsonlines
LOCK='configs/r17_PAIRED_CUE_PROBE_LOCK.json'


def save(path,value):
    p=ROOT/path;assert p.name.startswith('r17_') and p.parent.is_dir() and p.resolve().is_relative_to(ROOT.resolve())
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)


def gate():
    spec=read(LOCK);assert spec['status']=='frozen_observational_probe' and spec['new_method_candidates']==0 and not spec['heldout_access_authorized']
    for p,h in spec['source_sha256'].items():assert sha('scripts/'+p)==h,('Source changed',p)
    for p,h in spec['input_sha256'].items():assert sha(p)==h,('Input changed',p)
    public=read('logs/r17_PUBLIC_DEFINITION_RECEIPT.json');assert public['readback_verified'] and public['protocol_sha256']==sha(LOCK) and len(public['commit'])==40
    assert len(spec['cohort'])==len({c['id'] for c in spec['cohort']})==12
    return spec


def inputs(spec):
    authority={r['id']:r for s in ['discovery','selection'] for r in jsonlines('results/r7_'+s+'_development.jsonl')}
    manifests={r['id']:r for s in ['discovery','selection'] for r in jsonlines('data/slcd_'+s+'.jsonl')};states={r['id']:r for r in read('results/slcd_SOURCE_AUDIT.json')['states']};out=[]
    for chosen in spec['cohort']:
        qid=chosen['id'];q=manifests[qid];old=authority[qid];st=states[qid];j=chosen['candidate']
        assert old['predictions']['M0_R6']==j and q['prompt_sha256']==old['prompt_sha256'] and sha(st['file'])==st['sha256']
        req=q['request_pairs'][j];assert req[0].endswith('\nAnswer:')
        new=[req[0][:-len('Answer:')]+'Response:',req[1]]
        clean=dict(id=qid,request=req,changed_request=new,characters=q['choice_char_lengths'][j])
        expected=dict(main_sums={k:v['sum_scores'][j] for k,v in old['original_methods'].items()},reference_sums={k:v['sum_scores'][j] for k,v in old['prior_methods'][0].items()})
        out.append((chosen,clean,expected,st))
    return out
