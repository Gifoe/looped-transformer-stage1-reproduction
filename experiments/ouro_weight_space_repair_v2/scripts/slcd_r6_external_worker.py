"""Conditionally authorized joint R1/R6 external evaluation; original weights.

No downloads or task preparation. The parent supplies locked official zero-shot
manifests. Old main rows and ARC prior cache are read-only authorities. Fresh
external priors have this worker's own signature and a separate cache file.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
from slcd_core import ROOT,read,rows,sha,digest,save,writecsv
from slcd_r6_scores import selected_score

PRIMARY='r6_prior_projection'
PREFIX='Answer:'
BASE={f'native_k{k}' for k in range(1,7)}|{'loopcd_k4_fixed','loopcd_k4_adaptive'}
METHODS=('native4','fixed','adaptive','A1_g1p0',PRIMARY)
LOCK_PATH='configs/slcd_R6_EXTERNAL_LOCK.json'


def require(condition,message):
    if not condition:raise ValueError(message)


def path_of(relative):
    p=(ROOT/relative).resolve()
    require(p.is_relative_to(ROOT.resolve()),'Path escapes experiment root')
    return p


def append(relative,value):
    p=path_of(relative)
    require(p.parent.is_dir() and p.name.startswith('slcd_r6_'),'Invalid new output path')
    with p.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def validate_original(methods,lengths,gold):
    require(set(methods)==BASE,'Expected all eight original readouts')
    for value in methods.values():
        sums=value['sum_scores']
        require(len(sums)==len(lengths) and all(math.isfinite(x) for x in sums),'Invalid candidate sums')
        normalized=[s/max(n,1) for s,n in zip(sums,lengths)]
        pred=max(range(len(sums)),key=normalized.__getitem__)
        require(normalized==value['normalized_scores'] and pred==value['prediction']
                and value['correct']==(pred==gold),'Original readout normalization/prediction differs')


def joint_methods(original,prior,gold):
    arrays={name:np.asarray(value['normalized_scores'],dtype=np.float64) for name,value in original.items()}
    fixed=arrays['loopcd_k4_fixed']
    values={'native4':arrays['native_k4'],'fixed':fixed,'adaptive':arrays['loopcd_k4_adaptive'],
            'A1_g1p0':fixed+(arrays['native_k4']-arrays['native_k5']),
            PRIMARY:selected_score(original,prior,PRIMARY)}
    result={}
    for name,value in values.items():
        require(np.isfinite(value).all(),'Nonfinite ranking score')
        pred=int(value.argmax())
        result[name]=dict(normalized_scores=value.tolist(),prediction=pred,correct=pred==gold)
    return result


def paired_stats(good,baseline,rng):
    from scipy.stats import binomtest
    good=np.asarray(good,dtype=bool);baseline=np.asarray(baseline,dtype=bool)
    n=len(good);rescues=int((good&~baseline).sum());damage=int((baseline&~good).sum())
    draws=rng.multinomial(n,[rescues/n,damage/n,1-(rescues+damage)/n],size=50000)
    delta=100*(draws[:,0]-draws[:,1])/n
    low=float(np.quantile(delta,.025));high=float(np.quantile(delta,.975))
    return dict(n=n,correct=int(good.sum()),accuracy=float(good.mean()),corrections=rescues,
        regressions=damage,net_correct=rescues-damage,delta_accuracy_pp=100*(rescues-damage)/n,
        delta_pp=100*(rescues-damage)/n,exact_mcnemar_p=1. if rescues+damage==0 else float(binomtest(rescues,rescues+damage,.5).pvalue),
        paired_bootstrap_95ci_low_pp=low,paired_bootstrap_95ci_high_pp=high,
        paired_bootstrap_low_pp=low,paired_bootstrap_high_pp=high,bootstrap_replicates=50000)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['hellaswag','openbookqa'],required=True)
    args=parser.parse_args();started=time.time();task=args.task
    lock=read(LOCK_PATH);lock_sha=sha(LOCK_PATH)
    final_path='configs/slcd_R6_FINAL_LOCK.json';final=read(final_path);final_sha=sha(final_path)
    acceptance=read('results/slcd_R6_TEST_ACCEPTANCE.json');gate=read('results/slcd_R6_ARC_COMPLETE.json')
    require(sha('results/slcd_R6_TEST_ACCEPTANCE.json')==lock['arc_acceptance_sha256']
            and sha('results/slcd_R6_ARC_COMPLETE.json')==lock['arc_result_sha256'],'ARC gate receipt bytes changed')
    require(lock['expected_final_sha']==final_sha and lock['primary']==final['primary']==PRIMARY,'External primary/final binding differs')
    require(acceptance['status']=='passed' and acceptance['n']==1172 and acceptance['primary']==PRIMARY
            and acceptance['target_reached'] and acceptance['correct']>=733
            and acceptance['final_lock_sha256']==final_sha,'R6 acceptance gate has not passed')
    require(gate['status']=='complete' and gate['n']==1172 and gate['primary']==PRIMARY and gate['correct']>=733
            and gate['correct']==acceptance['correct'] and gate['final_lock_sha256']==final_sha,'R6 full ARC gate has not passed')
    require(sha('scripts/slcd_r6_external_worker.py')==lock['worker_source_sha256'],'External worker changed')
    require(sha('scripts/slcd_r6_scores.py')==lock['ranking_source_sha256']==final['ranking_source_sha256'],'Ranking kernel changed')
    for name,value in lock['inference_sources'].items():require(sha('scripts/'+name)==value,'Inference helper changed: '+name)
    required={'runtime.py','ouro_cached_scoring.py','slcd_core.py','slcd_token_worker_v2.py','weight_edits.py','slcd_r6_prior_worker.py'}
    require(required<=set(lock['inference_sources']),'External lock omits required helper')
    versions={name:importlib.metadata.version(name) for name in lock['package_versions']}
    require(versions==lock['package_versions'],'Evaluation environment changed')
    from runtime import encode_pair,outcome,check_budget,load_model
    from slcd_token_worker_v2 import file_sha,MODEL_SHA,parity_errors
    check_budget(120)
    model_dir=ROOT/'models/Ouro-1.4B';identity=final['model_identity']
    require(file_sha(model_dir/'model.safetensors')==identity['safetensors_sha256']==MODEL_SHA,'Original checkpoint differs')
    for name,value in identity['metadata_and_tokenizer_sha256'].items():require(file_sha(model_dir/name)==value,'Model/tokenizer metadata differs')
    spec=lock['tasks'][task];expected_n={'hellaswag':10042,'openbookqa':500}[task]
    require(spec['n']==expected_n and sha(spec['manifest_path'])==spec['manifest_sha256'],'Manifest count/hash differs')
    data=rows(spec['manifest_path']);expected={r['id']:r for r in data}
    require(len(data)==len(expected)==expected_n,'Manifest ID count differs')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    pairs_by_id={};prior_pairs_by_id={}
    for index,row in enumerate(data):
        if index%100==0:check_budget(120)
        require(row['id'].startswith(task+'/'),'Wrong task ID')
        require(hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256'],'Prompt SHA differs')
        require(len(row['request_pairs'])==len(row['choice_char_lengths'])==4 and 0<=row['gold']<4,'Malformed task choices')
        require(all(p[0]==row['prompt'] for p in row['request_pairs']),'Candidate prompt differs')
        require(all(isinstance(n,int) and n>0 for n in row['choice_char_lengths']),'Invalid character denominator')
        pairs_by_id[row['id']]=[encode_pair(tok,*p) for p in row['request_pairs']]
        prior_pairs_by_id[row['id']]=[encode_pair(tok,PREFIX,p[1]) for p in row['request_pairs']]

    old_main={};old_authority=None
    if spec.get('old_main_path'):
        require(sha(spec['old_main_path'])==spec['old_main_sha256'],'Old main snapshot differs')
        require(sha(spec['old_protocol_path'])==spec['old_protocol_sha256'],'Old protocol differs')
        protocol=read(spec['old_protocol_path']);old_final=read('FINAL_METHOD_LOCK.json')
        require(protocol['manifest_sha256']==spec['manifest_sha256'] and protocol['n']==expected_n
                and protocol['num_fewshot']==0 and protocol['harness_version']=='0.4.9','Old task protocol incompatible')
        old_worker_sha=sha('scripts/slcd_external_worker.py');old_final_sha=sha('FINAL_METHOD_LOCK.json')
        require(protocol['worker_source_sha256']==old_worker_sha and protocol['final_method_lock_sha256']==old_final_sha,'Old source/final receipt differs')
        expected_old_worker=spec.get('old_worker_source_sha256',lock.get('old_worker_source_sha256'))
        expected_old_final=spec.get('old_first_final_lock_sha256',lock.get('old_first_final_lock_sha256'))
        if expected_old_worker is not None:require(expected_old_worker==old_worker_sha,'Locked old worker differs')
        if expected_old_final is not None:require(expected_old_final==old_final_sha,'Locked original final differs')
        require(old_final['model_safetensors_sha256']==MODEL_SHA,'Old model differs')
        require(all(importlib.metadata.version(k)==v for k,v in old_final['environment'].items()),'Old environment differs')
        for name,value in old_final['tokenizer_and_model_metadata_sha256'].items():require(file_sha(model_dir/name)==value,'Old tokenizer/model metadata differs')
        for name,value in old_final['inference_source_sha256'].items():require(sha('scripts/'+name)==value,'Old inference source differs')
        old_values=rows(spec['old_main_path']);old_main={r['id']:r for r in old_values}
        require(len(old_values)==len(old_main) and set(old_main)<=set(expected),'Invalid old main IDs')
        for qid,value in old_main.items():
            row=expected[qid]
            require(value['protocol_sha256']==spec['old_protocol_sha256'] and value['primary']==old_final['primary'],'Old row authority differs')
            for key in ('gold','prompt','prompt_sha256','request_pairs','choice_char_lengths'):require(value[key]==row[key],'Old row input differs: '+key)
            counts=[len(t) for _,t in pairs_by_id[qid]]
            if 'choice_token_counts' in value:require(value['choice_token_counts']==counts,'Old token counts differ')
            validate_original(value['original_methods'],row['choice_char_lengths'],row['gold'])
        old_authority=dict(file=spec['old_main_path'],sha256=spec['old_main_sha256'],protocol_sha256=spec['old_protocol_sha256'],
            old_worker_source_sha256=old_worker_sha,old_final_lock_sha256=old_final_sha,
            token_count_binding='reconstructed_from_source_bound_request_pairs; compare recorded counts when present')

    # Authenticate the read-only ARC prior entries against its completed row index.
    prior_receipt_path='results/slcd_r6_priors_test_COMPLETE.json';prior_receipt=read(prior_receipt_path)
    old_provenance=prior_receipt['provenance'];old_global=old_provenance['global_cache_signature']
    require(prior_receipt['status']=='complete' and prior_receipt['n']==1172
            and old_provenance['authorization_lock_sha256']==final_sha,'ARC prior authority incomplete')
    require(old_provenance['model_identity']==identity,'ARC prior model differs')
    require(all(importlib.metadata.version(k)==v for k,v in old_provenance['package_versions'].items()),'ARC prior environment differs')
    for name,value in old_provenance['inference_source_sha256'].items():require(sha('scripts/'+name)==value,'ARC prior source differs')
    reconstructed=digest(dict(model_identity=identity,implementation=old_provenance['inference_source_sha256'],package_versions=old_provenance['package_versions'],
        prefix=PREFIX,seed=2021,scoring='full target LL raw sums; original BF16 and FP32 logsoftmax'))
    require(old_global==reconstructed and old_provenance['prior_prefix']==PREFIX,'ARC cache global signature differs')
    old_score_path=prior_receipt['score_file'];require(sha(old_score_path)==prior_receipt['score_file_sha256'],'ARC prior row authority changed')
    approved={}
    authority_rows=rows(old_score_path)
    require(len(authority_rows)==len({r['id'] for r in authority_rows})==1172,'ARC prior authority row coverage differs')
    for row in authority_rows:
        require(row['signature']==prior_receipt['signature'],'ARC prior row signature differs')
        require(len(row['cache_keys'])==len(row['cache_entry_sha256']),'ARC prior cache index lengths differ')
        for key,value in zip(row['cache_keys'],row['cache_entry_sha256']):
            require(key not in approved or approved[key]==value,'Conflicting ARC cache digests');approved[key]=value
    old_cache_path='results/slcd_r6_prior_cache.jsonl';old_cache_sha=sha(old_cache_path)
    old_cache={}
    def validate_entry(entry,global_signature):
        require(entry['signature']==global_signature and entry['prefix']==PREFIX and set(entry['sum_scores'])==BASE,'Cache signature/method set differs')
        require(all(math.isfinite(x) for x in entry['sum_scores'].values()),'Nonfinite prior sum')
        require(encode_pair(tok,PREFIX,entry['continuation'])==(entry['context_ids'],entry['target_ids']),'Cache tokenization differs')
        require(entry['key']==digest(dict(global_signature=global_signature,prefix=PREFIX,continuation=entry['continuation'],
                context_ids=entry['context_ids'],target_ids=entry['target_ids'])),'Prior cache key differs')
    def input_key(continuation,context,target):return digest(dict(prefix=PREFIX,continuation=continuation,context_ids=context,target_ids=target))
    for index,entry in enumerate(rows(old_cache_path)):
        if index%100==0:check_budget(120)
        validate_entry(entry,old_global)
        require(entry['key'] in approved and digest(entry)==approved[entry['key']],'ARC cache entry lacks completed authority')
        key=input_key(entry['continuation'],entry['context_ids'],entry['target_ids'])
        require(key not in old_cache,'Duplicate old prior input');old_cache[key]=entry
    raw_signature=digest(dict(external_lock_sha256=lock_sha,model_identity=identity,worker_source_sha256=lock['worker_source_sha256'],
        inference_sources=lock['inference_sources'],package_versions=versions,prefix=PREFIX,seed=2021,
        scoring='original scorer six-loop raw complete-target LL; no character normalization in cache'))
    cache_path='results/slcd_r6_external_prior_cache.jsonl';new_cache={}
    if path_of(cache_path).exists():
        for index,entry in enumerate(rows(cache_path)):
            if index%100==0:check_budget(120)
            validate_entry(entry,raw_signature)
            require(entry['origin']=='fresh_external_original_model' and entry['key'] not in new_cache,'New prior cache origin/duplicate differs')
            new_cache[entry['key']]=entry
    provenance=dict(external_lock_sha256=lock_sha,final_lock_sha256=final_sha,
        acceptance_sha256=sha('results/slcd_R6_TEST_ACCEPTANCE.json'),arc_gate_sha256=sha('results/slcd_R6_ARC_COMPLETE.json'),
        manifest_sha256=spec['manifest_sha256'],old_main_authority=old_authority,
        old_prior_authority=dict(receipt_sha256=sha(prior_receipt_path),score_sha256=prior_receipt['score_file_sha256'],cache_sha256=old_cache_sha,global_signature=old_global),
        global_external_prior_signature=raw_signature,model_identity=identity,package_versions=versions)
    signature=digest(provenance);stem='results/slcd_r6_'+task
    output=stem+'_scores.jsonl';completion=stem+'_COMPLETE.json'

    def entry_for_ref(reference):
        if reference['origin']=='old_arc_prior_authority':
            entry=old_cache[reference['input_key']]
        else:
            require(reference['origin']=='fresh_external_original_model','Unknown prior origin');entry=new_cache[reference['cache_key']]
        require(entry['key']==reference['cache_key'] and digest(entry)==reference['entry_sha256'],'Saved prior entry differs')
        return entry

    def validate_saved(value):
        require(value['id'] in expected and value['signature']==signature,'Saved row signature/ID differs')
        row=expected[value['id']]
        for key in ('gold','prompt','prompt_sha256','request_pairs','choice_char_lengths'):require(value[key]==row[key],'Saved row input differs: '+key)
        main_pairs=pairs_by_id[row['id']];prior_pairs=prior_pairs_by_id[row['id']]
        require(value['choice_token_counts']==[len(t) for _,t in main_pairs] and value['prior_token_counts']==[len(t) for _,t in prior_pairs],'Saved target counts differ')
        require(value['target_ids_equal_by_choice']==[t==pt for (_,t),(_,pt) in zip(main_pairs,prior_pairs)],'Saved target-boundary diagnostic differs')
        validate_original(value['original_methods'],row['choice_char_lengths'],row['gold'])
        if value['main_origin']=='old_authority':
            require(row['id'] in old_main and value['original_methods']==old_main[row['id']]['original_methods'],'Old main reuse changed')
        else:require(value['main_origin']=='fresh','Invalid main origin')
        require(len(value['prior_cache_entries'])==len(prior_pairs),'Saved prior count differs')
        entries=[entry_for_ref(ref) for ref in value['prior_cache_entries']]
        for entry,pair,request in zip(entries,prior_pairs,row['request_pairs']):
            require(entry['context_ids']==pair[0] and entry['target_ids']==pair[1] and entry['continuation']==request[1],'Saved prior input differs')
        prior={name:outcome([e['sum_scores'][name] for e in entries],row['choice_char_lengths'],row['gold']) for name in BASE}
        require(value['prior_methods']==prior and value['methods']==joint_methods(value['original_methods'],prior,row['gold']),'Saved prior/method scores differ')

    done={}
    if path_of(output).exists():
        values=rows(output);done={r['id']:r for r in values};require(len(values)==len(done) and set(done)<=set(expected),'Duplicate/unexpected saved rows')
        for index,value in enumerate(values):
            if index%100==0:check_budget(120)
            validate_saved(value)
    if path_of(completion).exists():
        old=read(completion);require(old['signature']==signature and old['n']==expected_n and set(done)==set(expected)
            and old['score_file_sha256']==sha(output),'Completed evaluation differs')
        for name,value in old['artifact_sha256'].items():require(sha(name)==value,'Completed artifact differs')
        print('R6_EXTERNAL_ALREADY_COMPLETE',task,flush=True);return
    model=scorer=None;head_sha=None
    def get_scorer():
        nonlocal model,scorer,head_sha
        if scorer is None:
            check_budget(120)
            from ouro_cached_scoring import OuroCachedScorer
            from weight_edits import tensor_sha256
            _,model=load_model();scorer=OuroCachedScorer(model,6);head_sha=tensor_sha256(model.lm_head.weight)
            require(head_sha==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256'],'Original head identity differs')
        return scorer

    # Independently replay three old saved main trajectories before reusing
    # their source-bound cohort; coverage is reported as three, never1956.
    old_ids=[row['id'] for row in data if row['id'] in old_main]
    probe_ids=[] if not old_ids else list(dict.fromkeys([old_ids[0],old_ids[len(old_ids)//2],old_ids[-1]]))
    for qid in probe_ids:
        check_budget(120);values=get_scorer().score_uncached(pairs_by_id[qid],return_states=False)
        direct={f'native_k{k}':dict(sum_scores=[v['sum_loglikelihood'][k] for v in values]) for k in range(1,7)}
        direct.update({f'loopcd_k4_{name}':dict(sum_scores=[v['loopcd'][name] for v in values]) for name in ('fixed','adaptive')})
        parity_errors(direct,dict(methods=old_main[qid]['original_methods']),4)
    replay_path=stem+'_BASELINE_REPLAY.json'
    save(replay_path,dict(status='passed',ids=probe_ids,n=len(probe_ids),eight_readouts_exact=True,max_sum_score_error=0.,
        signature=signature,old_main_reuse_coverage=len(old_main),coverage_scope='Only listed independently replayed rows; all other saved rows input/source bound'))

    for row in data:
        if row['id'] in done:continue
        check_budget(120);begin=time.time();qid=row['id'];pairs=pairs_by_id[qid]
        if qid in old_main:original=old_main[qid]['original_methods'];main_origin='old_authority'
        else:
            values=get_scorer().score_uncached(pairs,return_states=False)
            original={f'native_k{k}':outcome([v['sum_loglikelihood'][k] for v in values],row['choice_char_lengths'],row['gold']) for k in range(1,7)}
            original.update({f'loopcd_k4_{name}':outcome([v['loopcd'][name] for v in values],row['choice_char_lengths'],row['gold']) for name in ('fixed','adaptive')})
            main_origin='fresh'
        entries=[];references=[]
        for request,(context,target) in zip(row['request_pairs'],prior_pairs_by_id[qid]):
            check_budget(120);continuation=request[1];ik=input_key(continuation,context,target)
            if ik in old_cache:entry=old_cache[ik];origin='old_arc_prior_authority'
            else:
                key=digest(dict(global_signature=raw_signature,prefix=PREFIX,continuation=continuation,context_ids=context,target_ids=target))
                if key not in new_cache:
                    value=get_scorer().score_uncached([(context,target)],return_states=False)[0]
                    sums={f'native_k{k}':float(value['sum_loglikelihood'][k]) for k in range(1,7)}
                    sums.update({f'loopcd_k4_{name}':float(value['loopcd'][name]) for name in ('fixed','adaptive')})
                    entry=dict(key=key,signature=raw_signature,prefix=PREFIX,continuation=continuation,context_ids=context,target_ids=target,
                        sum_scores=sums,origin='fresh_external_original_model')
                    validate_entry(entry,raw_signature);append(cache_path,entry);new_cache[key]=entry
                entry=new_cache[key];origin='fresh_external_original_model'
            entries.append(entry);references.append(dict(origin=origin,input_key=ik,cache_key=entry['key'],entry_sha256=digest(entry)))
        prior={name:outcome([entry['sum_scores'][name] for entry in entries],row['choice_char_lengths'],row['gold']) for name in BASE}
        value=dict(row)
        value.update(choice_token_counts=[len(t) for _,t in pairs],prior_token_counts=[len(t) for _,t in prior_pairs_by_id[qid]],
            target_ids_equal_by_choice=[t==pt for (_,t),(_,pt) in zip(pairs,prior_pairs_by_id[qid])],
            main_origin=main_origin,original_methods=original,prior_methods=prior,prior_cache_entries=references,
            methods=joint_methods(original,prior,row['gold']),primary=PRIMARY,signature=signature,weights_modified=False,
            actual_shared_depth=6,elapsed_seconds=time.time()-begin)
        validate_saved(value);append(output,value);done[qid]=value
        if len(done)%100==0 or len(done)==expected_n:
            progress=dict(task=task,completed=len(done),target=expected_n,elapsed_seconds=time.time()-started)
            save('results/slcd_r6_EXTERNAL_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    require(set(done)==set(expected),'Incomplete task coverage')
    require(sha(LOCK_PATH)==lock_sha and sha(final_path)==final_sha and sha(spec['manifest_path'])==spec['manifest_sha256'],'Lock/manifest changed during evaluation')
    require(sha('scripts/slcd_r6_external_worker.py')==lock['worker_source_sha256'] and sha('scripts/slcd_r6_scores.py')==lock['ranking_source_sha256'],'Worker/ranking source changed')
    for name,value in lock['inference_sources'].items():require(sha('scripts/'+name)==value,'Inference source changed during evaluation')
    require(sha(old_cache_path)==old_cache_sha,'Read-only ARC prior cache changed')
    if old_authority:require(sha(spec['old_main_path'])==spec['old_main_sha256'],'Read-only old main changed')
    if model is not None:
        from weight_edits import tensor_sha256
        require(tensor_sha256(model.lm_head.weight)==head_sha,'Head changed during evaluation')
    rng=np.random.default_rng(2026);comparisons=[];transitions=[];full=[]
    for name in METHODS:
        good=[done[r['id']]['methods'][name]['correct'] for r in data]
        for baseline in ('fixed','A1_g1p0'):
            comparisons.append(dict(method=name,baseline=baseline,**paired_stats(good,[done[r['id']]['methods'][baseline]['correct'] for r in data],rng)))
        for row in data:
            value=done[row['id']];method=value['methods'][name];fixed=value['methods']['fixed'];a1=value['methods']['A1_g1p0']
            transitions.append(dict(method=name,id=row['id'],gold=row['gold'],prediction=method['prediction'],correct=method['correct'],
                fixed_prediction=fixed['prediction'],incumbent_prediction=a1['prediction'],base_correct=fixed['correct'],incumbent_correct=a1['correct']))
            for choice,score in enumerate(method['normalized_scores']):
                full.append(dict(id=row['id'],method=name,choice=choice,normalized_score=score,gold=row['gold'],prediction=method['prediction'],
                    correct=method['correct'],main_origin=value['main_origin'],choice_char_length=row['choice_char_lengths'][choice],
                    contextual_target_tokens=value['choice_token_counts'][choice],prior_target_tokens=value['prior_token_counts'][choice]))
    results=[r for r in comparisons if r['baseline']=='fixed']
    artifacts={stem+'_results.csv':results,stem+'_paired_comparisons.csv':comparisons,
               stem+'_paired_transitions.csv':transitions,stem+'_full_scores.csv':full}
    for path,values in artifacts.items():writecsv(path,values)
    main_old=sum(r['main_origin']=='old_authority' for r in done.values())
    old_prior_count=sum(ref['origin']=='old_arc_prior_authority' for r in done.values() for ref in r['prior_cache_entries'])
    receipt=dict(status='complete',task=task,n=expected_n,primary=PRIMARY,signature=signature,provenance=provenance,
        results=results,paired_comparisons=comparisons,score_file=output,score_file_sha256=sha(output),
        artifact_sha256={path:sha(path) for path in artifacts},final_lock_sha256=final_sha,external_lock_sha256=lock_sha,
        old_main_reuse_coverage=main_old,fresh_main_coverage=expected_n-main_old,old_prior_cache_choice_hits=old_prior_count,
        independent_baseline_parity_coverage=len(probe_ids),baseline_replay_receipt_sha256=sha(replay_path),independent_prior_reference_coverage=0,
        prior_reference_status='No independent prior reference; read-only authenticated ARC cache or unchanged original scorer',
        actual_shared_depth=6,logical_depths=dict(context=5,prior=4),vocabulary_head_depths=dict(context=[1,4,5],prior=[1,4]),
        target_boundary_policy='Each context uses its own official TemplateLM encoded complete continuation; equal bytes/character denominator, no truncation',
        weights_modified=False,no_runtime_speed_claim=True,elapsed_seconds=time.time()-started)
    save(completion,receipt);print(json.dumps(results),flush=True)


if __name__=='__main__':main()
