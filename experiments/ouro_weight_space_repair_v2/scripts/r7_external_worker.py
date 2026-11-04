"""Complete external tasks only after ARC>736. Same frozen primary formula.

M1 legitimately reuses complete original candidate LL (no vocab projection is
needed) and the frozen N0 LL. M2/M3 recompute full-vocabulary main/ref states.
All new writes contain numerical outputs, no HellaSwag text or target IDs.
"""
import argparse,json,os,time
import numpy as np
from r7_core import ROOT,read,save,sha,digest,jsonlines,baseline_vectors,csvout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['hellaswag','openbookqa'],required=True);args=ap.parse_args();task=args.task;start=time.time()
    lock=read('configs/r7_EXTERNAL_LOCK.json');primary=lock['primary'];spec=lock['tasks'][task]
    arc=read('results/r7_ARC_COMPLETE.json');assert arc['n']==1172 and arc['correct']>736 and sha('results/r7_ARC_COMPLETE.json')==lock['arc_complete_sha256']
    assert read('results/r7_TEST_ACCEPTANCE.json')['status']=='passed' and sha('results/r7_TEST_ACCEPTANCE.json')==lock['acceptance_sha256']
    assert sha('configs/r7_FINAL_METHOD_LOCK.json')==lock['final_method_lock_sha256']
    import importlib.metadata
    for name,value in lock['package_versions'].items():assert importlib.metadata.version(name)==value
    for name,value in lock['inference_source_sha256'].items():assert sha('scripts/'+name)==value
    assert sha(spec['manifest_path'])==spec['manifest_sha256'] and sha(spec['old_scores_path'])==spec['old_scores_sha256']
    data=list(jsonlines(spec['manifest_path']));old={r['id']:r for r in jsonlines(spec['old_scores_path'])}
    assert len(data)==len(old)==spec['n'] and {r['id'] for r in data}==set(old)
    for row in data:
        for key in ['gold','prompt','prompt_sha256','request_pairs','choice_char_lengths']:assert row[key]==old[row['id']][key]
    signature=digest(dict(external_lock_sha256=sha('configs/r7_EXTERNAL_LOCK.json'),task=task,primary=primary))
    output=f'results/r7_{task}_scores.jsonl';completion=f'results/r7_{task}_COMPLETE.json';assert not (ROOT/completion).exists()
    done={}
    if (ROOT/output).exists():
        for r in jsonlines(output):assert r['signature']==signature and r['id'] not in done;done[r['id']]=r
    from runtime import load_model,encode_pair,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from r7_inference import infer_question,forward_states,head_logits,original_sums,methods_from_sums
    from r7_references import prefix_alignment
    from r7_math import candidate_scores
    import torch
    from weight_edits import tensor_sha256
    tok,model=load_model();scorer=OuroCachedScorer(model,5)
    identity=lock['model_identity'];assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for name,value in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==value
    head_before=tensor_sha256(model.lm_head.weight);norm_before=tensor_sha256(model.model.norm.weight)
    assert head_before==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    prompts=[spec['references'][key] for key in ['N0','N1','N2']]
    probe_ids={data[0]['id'],data[len(data)//2]['id'],data[-1]['id']};replayed=[];max_peak=0
    for row in data:
        if row['id'] in done:continue
        check_budget(45);begin=time.time();historical=old[row['id']];expected=baseline_vectors(historical['original_methods'],historical['prior_methods'])
        torch.cuda.reset_peak_memory_stats()
        if primary=='M1_multi_candidate':
            pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
            assert [len(t) for p,t in pairs]==historical['choice_token_counts']
            rp=[[encode_pair(tok,prompt,p[1]) for p in row['request_pairs']] for prompt in prompts]
            align=[prefix_alignment(tok,request,pair,[p[i] for p in rp]) for i,(request,pair) in enumerate(zip(row['request_pairs'],pairs))]
            fallback=any(m is None for m,reason in align)
            if row['id'] in probe_ids:
                for i,pair in enumerate(pairs):
                    sums=original_sums(head_logits(model,forward_states(scorer,pair,5)),pair[1])
                    for name,value in sums.items():assert value==historical['original_methods'][name]['sum_scores'][i]
                    sums=original_sums(head_logits(model,forward_states(scorer,rp[0][i],4)),rp[0][i][1])
                    for name,value in sums.items():assert value==historical['prior_methods'][name]['sum_scores'][i]
                replayed.append(row['id'])
            priors=[{k:{p:v[p] for p in ['sum_scores','normalized_scores']} for k,v in historical['prior_methods'].items()}]
            if not fallback:
                for j in [1,2]:
                    sums={}
                    for pair in rp[j]:
                        values=original_sums(head_logits(model,forward_states(scorer,pair,4)),pair[1])
                        for name,value in values.items():sums.setdefault(name,[]).append(value)
                    priors.append(methods_from_sums(sums,np.asarray(row['choice_char_lengths'])))
                cleanoriginal={k:{p:v[p] for p in ['sum_scores','normalized_scores']} for k,v in historical['original_methods'].items()}
                score,diagnostic=candidate_scores(cleanoriginal,priors)
            else:score=expected['M0_R6'];diagnostic=None
            inferred=dict(method_scores={primary:score.tolist()},baseline_scores={k:v.tolist() for k,v in expected.items()},
                original_methods=historical['original_methods'],prior_methods=priors,M1_diagnostic=diagnostic,
                alignment_fallback=fallback,alignment_fallback_by_method={primary:fallback},alignment_reasons=[reason for m,reason in align],
                token_diagnostics=[],main_origin='source_bound_historical_complete_candidate_LLs',N0_origin='source_bound_historical_complete_candidate_LLs',
                actual_main_depth=None,actual_reference_depth=4)
        else:
            inferred=infer_question(model,scorer,tok,{'request_pairs':row['request_pairs'],'choice_char_lengths':row['choice_char_lengths']},[primary],prompts)
            for name,values in expected.items():assert np.array_equal(values,inferred['baseline_scores'][name]),('External baseline exact parity',row['id'],name)
            for name,value in inferred['prior_methods'][0].items():assert value['sum_scores']==historical['prior_methods'][name]['sum_scores']
            inferred.update(main_origin='fresh_original_K5',N0_origin='fresh_original_K4',actual_main_depth=5,actual_reference_depth=4)
        predictions={name:int(np.argmax(values)) for name,values in {**inferred['baseline_scores'],**inferred['method_scores']}.items()}
        peak=int(torch.cuda.max_memory_allocated());max_peak=max(max_peak,peak)
        record=dict(id=row['id'],gold=row['gold'],signature=signature,prompt_sha256=row['prompt_sha256'],choice_char_lengths=row['choice_char_lengths'],
            predictions=predictions,correct={name:pred==row['gold'] for name,pred in predictions.items()},
            **inferred,elapsed_seconds=time.time()-begin,peak_cuda_allocated_bytes=peak)
        with (ROOT/output).open('a',encoding='utf8') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
        done[row['id']]=record
        if len(done)%50==0 or len(done)==len(data):
            progress=dict(task=task,completed=len(done),target=len(data),elapsed_seconds=time.time()-start)
            save(f'results/r7_{task}_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    assert len(done)==len(data)==spec['n']
    assert sum(r['correct']['M0_R6'] for r in done.values())=={'hellaswag':7259,'openbookqa':219}[task]
    assert tensor_sha256(model.lm_head.weight)==head_before and tensor_sha256(model.model.norm.weight)==norm_before
    from r7_evaluate import paired
    values=[done[r['id']] for r in data];table=[paired(values,name,bootstrap=True) for name in values[0]['predictions']]
    csvout(f'results/r7_{task}_full_results.csv',table)
    csvout(f'results/r7_{task}_paired_transitions.csv',[dict(id=r['id'],gold=r['gold'],primary_prediction=r['predictions'][primary],
        R6_prediction=r['predictions']['M0_R6'],primary_correct=r['correct'][primary],R6_correct=r['correct']['M0_R6']) for r in values])
    save(completion,dict(status='complete',task=task,n=len(data),primary=primary,correct=sum(r['correct'][primary] for r in values),signature=signature,
        score_file=output,score_file_sha256=sha(output),metrics=next(r for r in table if r['method']==primary),
        external_lock_sha256=sha('configs/r7_EXTERNAL_LOCK.json'),old_score_authority_sha256=spec['old_scores_sha256'],
        historical_main_LLs_reused=len(data) if primary=='M1_multi_candidate' else 0,independent_original_replay_ids=replayed,
        baseline_per_question_exact=True,elapsed_seconds=time.time()-start,peak_cuda_allocated_bytes=max_peak,KV_cache=False,
        actual_main_depth=None if primary=='M1_multi_candidate' else 5,actual_reference_depth=4,weights_modified=False,
        serialized_HellaSwag_text=False,serialized_target_token_ids=False))
    print('R7_EXTERNAL_COMPLETE',task,flush=True)


if __name__=='__main__':main()
