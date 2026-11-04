"""R7 authorized stages, exact state authority and durable per-question output."""
import argparse,json,time,os
from r7_core import ROOT,read,save,sha,digest,gated_rows,jsonlines

SOURCES=['r7_worker.py','r7_inference.py','r7_math.py','r7_references.py','r7_core.py','runtime.py','ouro_cached_scoring.py','slcd_r6_scores.py']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',required=True);ap.add_argument('--job',required=True)
    ap.add_argument('--limit',type=int);ap.add_argument('--methods',default='all');ap.add_argument('--checks',action='store_true');a=ap.parse_args()
    assert a.job.replace('_','').isalnum();started=time.time()
    registry=read('configs/r7_METHODS.json');names=list(registry['methods']) if a.methods=='all' else a.methods.split(',')
    assert set(names)<=set(registry['methods'])
    implementation={n:sha('scripts/'+n) for n in SOURCES}
    if a.split in ['confirmation','validation','test']:
        lock=read('configs/r7_FINAL_METHOD_LOCK.json' if a.split=='test' else 'configs/r7_CANDIDATE_LOCK.json')
        assert implementation==lock['inference_source_sha256'] and sha('configs/r7_METHODS.json')==lock['methods_sha256']
        if a.split=='validation':
            vlock=read('configs/r7_VALIDATION_LOCK.json');assert names==vlock['validation_candidates'] and vlock['candidate_lock_sha256']==sha('configs/r7_CANDIDATE_LOCK.json')
        if a.split=='test':assert names==[lock['primary']]
        assert sha('configs/r7_reference_manifest.json')==lock['reference_manifest_sha256']
    # Authorization and source checks precede all held-out row decoding.
    data=gated_rows(a.split)
    if a.limit:assert a.split=='discovery';data=data[:a.limit]
    output=f'results/r7_{a.split}_{a.job}.jsonl';receipt=output.replace('.jsonl','_COMPLETE.json')
    provenance=dict(split=a.split,data_sha256=sha('data/slcd_'+a.split+'.jsonl'),methods=names,
        registry_sha256=sha('configs/r7_METHODS.json'),inference_source_sha256=implementation,limit=a.limit,
        reference_manifest_sha256=sha('configs/r7_reference_manifest.json'))
    signature=digest(provenance);done={}
    if (ROOT/output).exists():
        for row in jsonlines(output):assert row['id'] not in done and row['signature']==signature;done[row['id']]=row
        assert set(done)<={r['id'] for r in data}
    assert not (ROOT/receipt).exists(),'Already completed job must not be rerun'
    authority={};reference={};oldprior={}
    if a.split in ['discovery','selection','confirmation']:
        statepath='results/slcd_tokens_confirmation_confirmation_COMPLETE.json' if a.split=='confirmation' else 'results/slcd_SOURCE_AUDIT.json'
        sr=read(statepath);assert sr['status'] in ['passed','complete']
        historical=read('configs/slcd_R2_DEVELOPMENT_LOCK.json')
        assert sha(statepath)==historical['authority_sha256'][a.split]
        assert sha('data/slcd_'+a.split+'.jsonl')==historical['train_data_sha256'][a.split]
        if a.split=='confirmation':assert sr['provenance']['data_sha256']==historical['train_data_sha256'][a.split]
        else:assert sr['split_audits'][a.split]['data_sha256']==historical['train_data_sha256'][a.split]
        authority={r['id']:r for r in sr['states'] if r.get('split',a.split)==a.split}
        assert {r['id'] for r in data}<=set(authority)
        refpath=f'results/slcd_r2_{a.split}_development.jsonl' if a.split!='confirmation' else 'results/slcd_r2_confirmation_confirmation.jsonl'
        # Older confirmation R2 job name may differ; original token receipt is
        # authoritative and stores the complete native readouts used to create states.
        if a.split=='confirmation':refpath='results/slcd_tokens_confirmation_confirmation.jsonl'
        rr=read(refpath.replace('.jsonl','_COMPLETE.json'));assert sha(refpath)==rr['score_file_sha256']
        original_rows={r['id']:r for r in jsonlines(refpath)}
        reference={qid:r['original_methods'] for qid,r in original_rows.items()}
        import itertools
        with (ROOT/'results/slcd_r5_priors_train.jsonl').open() as f:
            allowed=f if a.split=='confirmation' else itertools.islice(f,576)
            prior_rows={r['id']:r for r in map(json.loads,allowed)}
            oldprior={qid:r['prior_methods'] for qid,r in prior_rows.items()}
        for item in data:
            assert item['prompt_sha256']==__import__('hashlib').sha256(item['prompt'].encode()).hexdigest()
            for old in [original_rows[item['id']],prior_rows[item['id']]]:
                assert old['prompt_sha256']==item['prompt_sha256'] and old['choice_char_lengths']==item['choice_char_lengths'] and old['gold']==item['gold']
            assert all(pair[0]==item['prompt'] for pair in item['request_pairs'])
    elif a.split in ['validation','test']:
        refpath=f'results/slcd_original_{a.split}.jsonl'
        assert sha(refpath)==lock['baseline_reference_sha256'][a.split], 'Historical baseline changed before parsing'
        original_rows={r['id']:r for r in jsonlines(refpath)}
        reference={qid:r['methods'] for qid,r in original_rows.items()}
        assert set(reference)=={r['id'] for r in data}
        for item in data:
            old=original_rows[item['id']]
            assert old['prompt_sha256']==item['prompt_sha256'] and old['choice_char_lengths']==item['choice_char_lengths'] and old['gold']==item['gold']
    from runtime import load_model,check_budget
    import torch
    from ouro_cached_scoring import OuroCachedScorer
    from weight_edits import tensor_sha256
    from r7_inference import infer_question
    tok,model=load_model();scorer=OuroCachedScorer(model,5)
    identity=read('configs/slcd_R6_FINAL_LOCK.json')['model_identity']
    assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for name,value in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==value
    before=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight))
    assert before['head']==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    refs=read('configs/r7_reference_manifest.json')['references'];refprompts=[refs[k] for k in ['N0','N1','N2']]
    from lm_eval.utils import apply_template
    from r7_references import arc_references
    template=read('configs/r7_reference_manifest.json')['official_template']
    maxerror=0.;peakallocated=0
    for row in data:
        if row['id'] in done:continue
        check_budget(45);begin=time.time();states=None
        rendered,_=arc_references(row,lambda doc:apply_template(template,doc));assert rendered==refprompts,'Reference prefix differs from frozen fixed25 protocol'
        if authority:
            entry=authority[row['id']];assert sha(entry['file'])==entry['sha256']
            payload=torch.load(ROOT/entry['file'],map_location='cpu',weights_only=True);assert payload['id']==row['id'];states=payload['candidate_states']
        torch.cuda.reset_peak_memory_stats()
        clean={'request_pairs':row['request_pairs'],'choice_char_lengths':row['choice_char_lengths']}
        inferred=infer_question(model,scorer,tok,clean,names,refprompts,main_states=states,checks=a.checks and not done)
        errors={}
        if reference:
            for name,value in inferred['original_methods'].items():
                err=max(abs(x-y) for x,y in zip(value['sum_scores'],reference[row['id']][name]['sum_scores']));errors[name]=err;assert err==0.,('main parity',row['id'],name,err)
            if oldprior:
                for name,value in inferred['prior_methods'][0].items():
                    err=max(abs(x-y) for x,y in zip(value['sum_scores'],oldprior[row['id']][name]['sum_scores']));errors['N0_'+name]=err;assert err==0.,('N0 parity',row['id'],name,err)
            maxerror=max(maxerror,max(errors.values()))
        allscores={**inferred['baseline_scores'],**inferred['method_scores']}
        predictions={k:max(range(len(v)),key=v.__getitem__) for k,v in allscores.items()}
        peak=int(torch.cuda.max_memory_allocated());peakallocated=max(peakallocated,peak)
        record=dict(id=row['id'],gold=row['gold'],signature=signature,prompt_sha256=row['prompt_sha256'],
            choice_char_lengths=row['choice_char_lengths'],predictions=predictions,
            correct={k:p==row['gold'] for k,p in predictions.items()},**inferred,
            baseline_parity_errors=errors,elapsed_seconds=time.time()-begin,peak_cuda_allocated_bytes=peak)
        with (ROOT/output).open('a',encoding='utf8') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        done[row['id']]=record
        progress=dict(split=a.split,job=a.job,completed=len(done),target=len(data),elapsed_seconds=time.time()-started,
            correct={k:sum(r['correct'][k] for r in done.values()) for k in allscores})
        save(f'results/r7_{a.job}_PROGRESS.json',progress)
        if len(done)%20==0 or len(done)==len(data):print(json.dumps(progress),flush=True)
    after=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before==after
    assert implementation=={n:sha('scripts/'+n) for n in SOURCES}
    import resource
    final=dict(status='complete',n=len(data),signature=signature,provenance=provenance,score_file=output,score_file_sha256=sha(output),
        correct={k:sum(r['correct'][k] for r in done.values()) for k in next(iter(done.values()))['predictions']},
        alignment_fallbacks=sum(r['alignment_fallback'] for r in done.values()),
        head_norm_before=before,head_norm_after=after,weights_modified=False,
        baseline_parity_max_error=maxerror if reference else None,baseline_parity_questions=len(data) if reference else 0,
        baseline_parity_status='exact' if reference else 'not_available',
        elapsed_seconds=time.time()-started,peak_cuda_allocated_bytes=peakallocated,host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        logical_main_depth=5,actual_main_depth=None if authority else 5,historical_main_state_collection_depth=6 if authority else None,
        actual_reference_depth=4,kv_cache_used=False,diagnostic_model_hooks_used=False,
        historical_states_reused=bool(authority),diagnostic_overhead_included=True)
    save(receipt,final);print('R7_COMPLETE',json.dumps(final),flush=True)


if __name__=='__main__':main()
