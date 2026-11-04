"""Strict scope/source gate for one bounded post-evaluation mechanism probe."""
import json
from r7_core import ROOT,read,sha,digest,jsonlines

LOCK='configs/r15_CONTEXT_PROBE_LOCK.json'


def save(path,value):
    p=ROOT/path;assert p.name.startswith('r15_') and p.parent.is_dir() and p.resolve().is_relative_to(ROOT.resolve())
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)


def gate():
    spec=read(LOCK);assert spec['status']=='frozen_observational_probe' and spec['new_method_candidates']==0 and not spec['heldout_access_authorized']
    for name,h in spec['source_sha256'].items():assert sha('scripts/'+name)==h,('Source changed',name)
    for name,h in spec['input_sha256'].items():assert sha(name)==h,('Input changed',name)
    assert len(spec['cohort'])==58 and len({r['id'] for r in spec['cohort']})==58
    public=read('logs/r15_PUBLIC_DEFINITION_RECEIPT.json')
    assert public['readback_verified'] and public['protocol_sha256']==sha(LOCK) and len(public['commit'])==40
    return spec


def inputs(spec):
    # Only two development splits; no function in this collector opens heldout.
    manifests={};authority={}
    for split in ['discovery','selection']:
        for row in jsonlines('data/slcd_'+split+'.jsonl'):manifests[row['id']]=row
        for row in jsonlines('results/r7_'+split+'_development.jsonl'):authority[row['id']]=row
    statebook=read('results/slcd_SOURCE_AUDIT.json');states={r['id']:r for r in statebook['states']}
    refbook=read('configs/r7_reference_manifest.json');assert refbook['official_template']=='Question: {{question}}\nAnswer:'
    from r7_references import arc_references
    result=[]
    for chosen in spec['cohort']:
        qid=chosen['id'];item=manifests[qid];old=authority[qid];state=states[qid]
        rendered,_=arc_references(dict(source_doc={'question':item['question']},prompt=item['prompt']),lambda doc:'Question: '+doc['question']+'\nAnswer:')
        assert rendered==[refbook['references'][k] for k in ['N0','N1','N2']]
        assert sha(state['file'])==state['sha256'] and item['prompt_sha256']==old['prompt_sha256'] and old['baseline_parity_errors']
        assert max(old['baseline_parity_errors'].values())==0
        a=old['predictions']['M0_R6'];b=old['predictions']['M2_N0_token'];assert chosen['old_candidate']==a
        if chosen['group']=='disagreement':assert a!=b and chosen['probe_candidate']==b
        else:
            assert a==b
            rival=max((j for j in range(len(item['choices'])) if j!=a),key=lambda j:old['baseline_scores']['M0_R6'][j]);assert chosen['probe_candidate']==rival
        # Gold exists in inherited manifests but is never passed to measurement.
        clean=dict(id=qid,request_pairs=item['request_pairs'],choice_char_lengths=item['choice_char_lengths'])
        expected=dict(original_methods=old['original_methods'],prior_methods=old['prior_methods'],method_scores={'M2_N0_token':old['method_scores']['M2_N0_token']},
            token_diagnostics=old['token_diagnostics'])
        result.append((chosen,clean,expected,state))
    return result
