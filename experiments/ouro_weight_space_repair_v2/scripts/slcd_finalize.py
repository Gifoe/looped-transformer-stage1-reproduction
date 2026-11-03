"""Single validation selection and locked test; no adaptive held-out search."""
import argparse,csv,hashlib,json,shutil,time
import numpy as np
from slcd_core import ROOT,read,rows,sha,digest,save,writecsv,scores_for,summarize,add_iteration

BASE=['native4','fixed','adaptive']
SOURCES=['slcd_token_worker_v2.py','slcd_logit_methods.py','runtime.py','ouro_cached_scoring.py','slcd_core.py']
MANIFEST_SHA={'validation':'cb7cb3f8cf2bc80523f775a41961352647845d71c4a0c8f91cfcb11aaab1257a','test':'c21f826e316eac7d5101787aae1e4cea5515a820cc0f0d5d577582a7b9c93eb6'}

def immutable(path,value):
    if (ROOT/path).exists():assert read(path)==value,'Immutable artifact already differs: '+path
    else:save(path,value)

def closure(ids,registry):
    out={}
    def visit(mid):
        if mid in out:return
        out[mid]=registry[mid]
        for key in ['a','b','fallback']:
            if key in registry[mid]:visit(registry[mid][key])
    for mid in ids:visit(mid)
    return out

def recipes_for(ids,registry):
    subset=closure(ids,registry)
    recipes=[dict(id='tf_fixed',kind='fixed',omega=.5),dict(id='tf_adaptive',kind='adaptive',omega=1.)]
    for s in subset.values():
        if s['kind']=='token' and s['recipe_id'] not in {r['id'] for r in recipes}:recipes.append(s['token_recipe'])
    return dict(recipes=recipes,scope='only selected/frozen methods plus mandatory controls')

def token_rows(split,job):
    path=f'results/slcd_tokens_{split}_{job}.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'))
    assert receipt['status']=='complete' and sha(path)==receipt['score_file_sha256']
    data=rows(path);assert len(data)==receipt['n']==len({r['id'] for r in data})
    assert {r['id'] for r in data}=={r['id'] for r in rows(f'data/slcd_{split}.jsonl')}
    assert all(r['implementation_signature']==receipt['implementation_signature'] and r['data_sha256']==sha(f'data/slcd_{split}.jsonl') for r in data)
    if split in ['validation','test']:
        lp='configs/slcd_VALIDATION_LOCK.json' if split=='validation' else 'FINAL_METHOD_LOCK.json';lock=read(lp);p=receipt['provenance']
        assert p['recipes_sha256'] in lock['allowed_recipe_sha256'] and p['data_sha256']==lock['dataset_sha256'][split]
        assert p['authorization_lock_sha256']==sha(lp)
        for name,h in lock['inference_source_sha256'].items():assert sha('scripts/'+name)==h==p['implementation'][name]
    return {r['id']:r for r in data}

def dataset(split,tokens=None):
    source=rows(f'data/slcd_{split}.jsonl')
    originals={r['id']:r for r in rows(f'results/slcd_original_{split}.jsonl')} if (ROOT/f'results/slcd_original_{split}.jsonl').exists() else tokens
    assert originals is not None and set(originals)=={r['id'] for r in source}
    result=[]
    for r in source:
        s=originals[r['id']];assert r['gold']==s['gold'] and r['prompt_sha256']==s['prompt_sha256'] and r['choice_char_lengths']==s['choice_char_lengths']
        if 'original_methods' not in s:
            assert s['data_sha256']==sha(f'data/slcd_{split}.jsonl')
            if split in ['validation','test']:
                assert s['fewshot']==25 and s['demonstrations']=='fixed_global_25' and s['cached'] is False
        methods=s.get('original_methods',s['methods']);m={}
        for name in [f'native_k{k}' for k in range(1,7)]+['loopcd_k4_fixed','loopcd_k4_adaptive']:
            v=methods[name];x=np.asarray(v['sum_scores'],dtype=float)/np.maximum(r['choice_char_lengths'],1)
            assert np.array_equal(x,np.asarray(v['normalized_scores'])) and int(x.argmax())==v['prediction'] and (int(x.argmax())==r['gold'])==v['correct']
            m[name]=x
        result.append(dict(id=r['id'],gold=r['gold'],methods=m,source=r,saved=s))
    return result

def summarize_ids(split,ids,reg,tokens=None):
    data=dataset(split,tokens);out=[]
    for mid in ids:
        s=summarize(data,reg[mid],reg,tokens);s.pop('outcomes')
        out.append(dict(split=split,method=mid,spec_sha256=digest(reg[mid]),**s))
    return out

def confirmation():
    lock=read('configs/slcd_CONFIRMATION_METHOD_LOCK.json');reg=read('configs/slcd_METHODS.json')
    assert sha('configs/slcd_METHODS.json')==lock['registry_sha256']
    ts=token_rows('confirmation','confirmation')
    res=summarize_ids('confirmation',BASE+lock['methods'],reg,ts)
    immutable('results/slcd_iteration5.json',dict(status='complete',train_only=True,predeclared_method_lock_sha256=sha('configs/slcd_CONFIRMATION_METHOD_LOCK.json'),results=res))
    writecsv('results/slcd_confirmation_results.csv',res)
    print(json.dumps([{k:r[k] for k in ['method','correct','baseline_correct','corrections','regressions','net_correct']} for r in res],indent=2))

def freeze_validation():
    reg=read('configs/slcd_METHODS.json');c=read('configs/slcd_CONFIRMATION_METHOD_LOCK.json');assert sha('configs/slcd_METHODS.json')==c['registry_sha256']
    confirmed=read('results/slcd_iteration5.json')['results']
    policy=read('configs/slcd_ADVANCEMENT_POLICY.json')
    assert policy['frozen_before_confirmation_results_revealed'] and policy['candidate_pool']==c['methods']
    cr={r['method']:r for r in confirmed}
    finalists=sorted(c['methods'],key=lambda mid:(-cr[mid]['correct'],reg[mid]['required_depth'],reg[mid]['complexity'],mid))[:policy['advance_count']]
    recipes=recipes_for(finalists,reg);immutable('configs/slcd_TOKEN_VALIDATION.json',recipes)
    development=[]
    for i in range(1,6):
        name={1:'slcd_iteration1.json',2:'slcd_iteration2_token.json',3:'slcd_iteration3.json',4:'slcd_iteration4.json',5:'slcd_iteration5.json'}[i]
        development+=read('results/'+name)['results']
    development+=read('results/slcd_candidate_pilot.json')['results']
    evidence={}
    for mid in finalists:
        rs={r['split']:r for r in development if r['method']==mid}
        assert {'discovery','selection','confirmation'}<=set(rs),mid
        evidence[mid]={s:rs[s] for s in ['discovery','selection','confirmation']}
    obj=dict(status='frozen_before_official_validation_access',created_unix=time.time(),finalists=finalists,method_registry=closure(BASE+finalists,reg),
        confirmation_evidence=confirmed,development_evidence=evidence,advancement_policy_sha256=sha('configs/slcd_ADVANCEMENT_POLICY.json'),
        selection_rule=['largest validation correct count','largest minimum proportional net gain across discovery, selection, confirmation','fewest validation correctness-discordant pairs versus Fixed at tied net','lowest required logical loop depth','lowest registered complexity','lexicographically first method ID'],
        no_feedback_revision=True,one_primary=True,secondary=None,baseline_eligible_as_primary=False,
        allowed_recipe_sha256=[sha('configs/slcd_TOKEN_VALIDATION.json')],dataset_sha256={'validation':MANIFEST_SHA['validation']},
        inference_source_sha256={n:sha('scripts/'+n) for n in SOURCES},controller_source_sha256=sha('scripts/slcd_finalize.py'),
        budget_limit_seconds=21600,historical_test_exposure='Previously used regression benchmark; retrospective A1 gamma1 726 not used for selection')
    assert not (ROOT/'configs/slcd_VALIDATION_LOCK.json').exists()
    save('configs/slcd_VALIDATION_LOCK.json',obj)
    stage('validation')
    print('VALIDATION_LOCKED',len(finalists),sha('configs/slcd_VALIDATION_LOCK.json'))

def stage(split):
    lock=read('configs/slcd_VALIDATION_LOCK.json' if split=='validation' else 'FINAL_METHOD_LOCK.json')
    assert lock['dataset_sha256'][split]==MANIFEST_SHA[split]
    old=ROOT.parent/'ouro_weight_space_repair_v1'
    targets=[(old/f'data/{split}.jsonl',ROOT/f'data/slcd_{split}.jsonl'),(old/f'results/frozen_{split}/scores.jsonl',ROOT/f'results/slcd_original_{split}.jsonl')]
    assert hashlib.sha256(targets[0][0].read_bytes()).hexdigest()==MANIFEST_SHA[split]
    for src,dst in targets:
        if dst.exists():assert hashlib.sha256(dst.read_bytes()).digest()==hashlib.sha256(src.read_bytes()).digest()
        else:shutil.copyfile(src,dst)

def select():
    lock=read('configs/slcd_VALIDATION_LOCK.json');assert sha('scripts/slcd_finalize.py')==lock['controller_source_sha256']
    reg=lock['method_registry'];ts=token_rows('validation','final')
    receipt=read('results/slcd_tokens_validation_final_COMPLETE.json');assert receipt['independent_parity_coverage']==299 and receipt['all_baseline_parity_exact']
    res=summarize_ids('validation',BASE+lock['finalists'],reg,ts);by={r['method']:r for r in res}
    def order(mid):
        r=by[mid];dev=lock['development_evidence'][mid]
        return (-r['correct'],-min(v['net_correct']/v['n'] for v in dev.values()),r['corrections']+r['regressions'],reg[mid]['required_depth'],reg[mid]['complexity'],mid)
    ranked=sorted(lock['finalists'],key=order);primary=ranked[0]
    immutable('results/slcd_validation_results.json',dict(status='complete',results=res,ranking=ranked,primary=primary,selection_rule=lock['selection_rule']))
    writecsv('results/slcd_validation_results.csv',res)
    recipes=recipes_for([primary],reg);immutable('configs/slcd_TOKEN_TEST.json',recipes)
    audit=read('results/slcd_SOURCE_AUDIT.json');selected=closure(BASE+[primary],reg)
    final=dict(status='frozen_final_method',created_unix=time.time(),primary=primary,secondary=None,method_registry=selected,coefficients_and_definitions=selected,
        model_checkpoint='ByteDance/Ouro-1.4B',model_revision=audit['model_revision'],model_safetensors_sha256=audit['model_safetensors_sha256'],weights_modified=False,
        tokenizer_and_model_metadata_sha256=audit['tokenizer_and_model_metadata_sha256'],environment=audit['package_versions'],
        prompt_protocol='Same global fixed25 ARC train demonstrations, seed2021, lm-eval0.4.9; complete candidate teacher-forced token likelihood, divided by answer character count; first argmax ties',
        fixed_demonstrations_sha256=sha('data/slcd_fixed_demonstrations.json'),seed=2021,dataset_revision_notice_sha256=sha('data/DATA_SOURCE_NOTICE.txt'),
        dataset_sha256={'test':MANIFEST_SHA['test']},allowed_recipe_sha256=[sha('configs/slcd_TOKEN_TEST.json')],
        inference_source_sha256={n:sha('scripts/'+n) for n in SOURCES},controller_source_sha256=sha('scripts/slcd_finalize.py'),
        validation_lock_sha256=sha('configs/slcd_VALIDATION_LOCK.json'),selection_evidence=res,ranking=ranked,selection_rule=lock['selection_rule'],
        final_test_access_authorized=True,no_post_test_changes=True,other_benchmark_gate=dict(primary_correct_strictly_greater_than=717,n=1172),
        logical_loop_depth=reg[primary]['required_depth'],shared_actual_trajectory_depth=6,runtime_speed_claim_permitted=False,
        test_status='Previously used regression benchmark, not blind. Retrospective A1 gamma1 726 was exposed before this research; not a selection criterion.')
    assert not (ROOT/'FINAL_METHOD_LOCK.json').exists();save('FINAL_METHOD_LOCK.json',final)
    stage('test')
    print(json.dumps(dict(primary=primary,validation=by[primary],test_requires_token_recipes=len(recipes['recipes'])>2,final_lock_sha256=sha('FINAL_METHOD_LOCK.json')),indent=2))

def test():
    lock=read('FINAL_METHOD_LOCK.json');assert lock['final_test_access_authorized'] and sha('scripts/slcd_finalize.py')==lock['controller_source_sha256']
    for n,h in lock['inference_source_sha256'].items():assert sha('scripts/'+n)==h
    assert sha('data/slcd_test.jsonl')==lock['dataset_sha256']['test']
    reg=lock['method_registry'];primary=lock['primary'];need=any(s['kind']=='token' for s in reg.values());ts=token_rows('test','final') if need else None
    if need:
        receipt=read('results/slcd_tokens_test_final_COMPLETE.json');assert receipt['independent_parity_coverage']==1172 and receipt['all_baseline_parity_exact']
    data=dataset('test',ts);assert len(data)==1172
    old=ROOT.parent/'ouro_weight_space_repair_v1';completion=json.loads((old/'results/frozen_test/COMPLETE.json').read_text())
    assert completion['status']=='complete' and completion['n']==completion['expected_n']==1172 and completion['data_sha256']==MANIFEST_SHA['test']
    codehash=hashlib.sha256(''.join(hashlib.sha256((old/'scripts'/n).read_bytes()).hexdigest() for n in ['runtime.py','ouro_cached_scoring.py','run_frozen_eval.py']).encode()).hexdigest()
    assert codehash==completion['implementation_sha256']
    assert all(r['saved']['implementation_sha256']==codehash and r['saved']['recipe_sha256']==completion['recipe_sha256'] for r in data)
    from transformers import AutoTokenizer
    from runtime import encode_pair
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Ouro-1.4B',local_files_only=True)
    for row in data:
        assert [len(encode_pair(tok,*p)[1]) for p in row['source']['request_pairs']]==row['saved']['choice_token_counts']
    immutable('results/slcd_BASELINE_REUSE_AUDIT.json',dict(status='passed',n=1172,original_recipe_sha256=completion['recipe_sha256'],old_implementation_sha256=codehash,
        full_prompt_candidate_gold_character_lengths_and_token_counts_equal=True,full_baseline_receipt_bound=True,
        current_source_audit_sha256=sha('results/slcd_SOURCE_AUDIT.json'),validation_replay_receipt_sha256=sha('results/slcd_tokens_validation_final_COMPLETE.json'),
        environment_is_shared_original_v1_virtualenv=True,test_replay_coverage=1172 if need else 0,weights_modified=False))
    res=[];trans=[];rng=np.random.default_rng(2021)
    for mid in BASE+[primary]:
        s=summarize(data,reg[mid],reg,ts);out=s.pop('outcomes');r=s['corrections'];d=s['regressions'];n=s['n']
        draws=rng.multinomial(n,[r/n,d/n,1-(r+d)/n],size=50000);delta=100*(draws[:,0]-draws[:,1])/n
        s.update(paired_bootstrap_95ci_low_pp=float(np.quantile(delta,.025)),paired_bootstrap_95ci_high_pp=float(np.quantile(delta,.975)),bootstrap_replicates=50000)
        res.append(dict(method=mid,**s))
        for row,o in zip(data,out):
            p=o['prediction'];bp=int(row['methods']['loopcd_k4_fixed'].argmax())
            trans.append(dict(method=mid,id=row['id'],gold=row['gold'],fixed_prediction=bp,prediction=p,transition='wrong_to_correct' if o['correct'] and not o['base_correct'] else 'correct_to_wrong' if o['base_correct'] and not o['correct'] else 'correct_to_correct' if o['correct'] else 'wrong_to_wrong',answer_changed=bp!=p,**{k:o[k] for k in ['correct','base_correct','margin','base_margin']}))
    by={r['method']:r for r in res};assert [by[k]['correct'] for k in BASE]==[702,717,716]
    result=dict(status='complete',n=1172,primary=primary,primary_correct=by[primary]['correct'],passed_absolute_gate=by[primary]['correct']>717,results=res,
        final_lock_sha256=sha('FINAL_METHOD_LOCK.json'),other_benchmarks_authorized=by[primary]['correct']>717,no_test_reselection=True,
        test_is_blind=False,baseline_reuse='Full prompts/character lengths/labels verified against historical baseline rows; original model SHA, environment, tokenizer, inference source bound by SOURCE_AUDIT; independent eight-baseline exact replay576 training questions and299 official validation questions. Token methods additionally exact replay1172 test questions; CPU score-only methods reuse compatible saved baseline sums.')
    immutable('results/slcd_ARC_TEST_COMPLETE.json',result);writecsv('results/slcd_arc_full_test_results.csv',res);writecsv('results/slcd_paired_transitions.csv',trans)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['confirmation','freeze-validation','select','test']);a=ap.parse_args()
    {'confirmation':confirmation,'freeze-validation':freeze_validation,'select':select,'test':test}[a.stage]()
