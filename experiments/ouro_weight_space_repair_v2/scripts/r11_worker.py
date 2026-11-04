"""Fresh auxiliary K4, selected-only outcomes, and actual standalone profiling."""
import argparse,json,os,time
import numpy as np
from r11_core import ROOT,read,sha,digest,jsonlines,save,gate,rows,extra_sources
from r11_inference import infer_auxiliary,select,joint_baselines,joint_prefix


def profile(model,scorer,tok,data,names,prefix,lock):
    import torch
    from runtime import encode_pair,check_budget
    from r7_inference import forward_states,head_logits,original_sums,methods_from_sums
    from r7_core import baseline_vectors
    from slcd_r6_scores import selected_score
    historical={r['id']:r for r in jsonlines('results/r11_development_scores.jsonl')};primary=names[0];robust=primary=='robust_cyclic_loop_axis'
    def fresh(row):
        sums={};priors={}
        for request in row['request_pairs']:
            q=head_logits(model,forward_states(scorer,encode_pair(tok,*request),5))
            p=head_logits(model,forward_states(scorer,encode_pair(tok,'Answer:',request[1]),4))
            for name,v in original_sums(q,encode_pair(tok,*request)[1]).items():sums.setdefault(name,[]).append(v)
            for name,v in original_sums(p,encode_pair(tok,'Answer:',request[1])[1]).items():priors.setdefault(name,[]).append(v)
        lengths=np.asarray(row['choice_char_lengths']);return methods_from_sums(sums,lengths),methods_from_sums(priors,lengths)
    actual=[];base=[]
    for value in data:
        check_budget(30);row=value['item'];torch.cuda.synchronize();begin=time.time();torch.cuda.reset_peak_memory_stats()
        original,prior=fresh(row)
        for new,old in [(original,value['original']),(prior,value['prior'])]:
            for k,v in new.items():assert np.array_equal(v['sum_scores'],old[k]['sum_scores'])
        aux=infer_auxiliary(model,scorer,tok,dict(question=row['question'],choices=row['choices']),prefix,robust)
        scores,_=select(original,prior,aux,names);assert np.array_equal(scores[primary],historical[row['id']]['method_scores'][primary])
        assert np.array_equal(selected_score(original,prior,'r6_prior_projection'),value['baseline_scores']['M0_R6'])
        torch.cuda.synchronize();actual.append(dict(id=row['id'],seconds=time.time()-begin,peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),score_max_error=0.))
    for value in data:
        check_budget(30);row=value['item'];torch.cuda.synchronize();begin=time.time();original,prior=fresh(row)
        assert np.array_equal(selected_score(original,prior,'r6_prior_projection'),value['baseline_scores']['M0_R6']);torch.cuda.synchronize()
        base.append(dict(id=row['id'],seconds=time.time()-begin,score_max_error=0.))
    warm=float(np.mean([x['seconds'] for x in actual[1:]]));reference=float(np.mean([x['seconds'] for x in base[1:]]))
    result=dict(status='complete',n=12,primary=primary,actual_main_depth=5,actual_prior_depth=4,actual_auxiliary_depth=4,
        robust_references=robust,KV_cache=False,attention_mask_argument=False,historical_states=False,cached_main_scores=False,
        diagnostic_hooks=False,warm_mean_question_seconds=warm,R6_warm_mean_question_seconds=reference,time_ratio=warm/reference,
        peak_cuda_allocated_bytes=max(x['peak_cuda_allocated_bytes'] for x in actual),records=actual,R6_records=base,
        scope='Same12 fixed discovery questions, warm11 averages. Sequential timing, small sample, no general hardware speed claim.',
        final_method_lock_sha256=sha('configs/r11_FINAL_METHOD_LOCK.json'),extra_source_sha256=extra_sources())
    save('results/r11_DEPLOYMENT_PROFILE.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['parity','confirmation','validation','profile','test'],required=True);a=p.parse_args();started=time.time()
    protocol,lock,lockpath,names=gate(a.phase);data=rows(a.phase)
    output=f'results/r11_{a.phase}_selected.jsonl';receipt=output.replace('.jsonl','_COMPLETE.json')
    assert not (ROOT/receipt).exists(),'Completed attempts are immutable'
    sig=digest(dict(phase=a.phase,protocol_sha256=sha('configs/r11_PROTOCOL.json'),lock_sha256=sha(lockpath),names=names,extra_source_sha256=extra_sources()))
    done={}
    if (ROOT/output).exists():
        for r in jsonlines(output):assert r['signature']==sig and r['id'] not in done;done[r['id']]=r
    assert set(done)<={v['item']['id'] for v in data}
    from runtime import load_model,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from weight_edits import tensor_sha256
    import torch
    tok,model=load_model();scorer=OuroCachedScorer(model,4);identity=protocol['model_identity']
    assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for n,h in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+n)==h
    before=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before['head']==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    assert sha('data/slcd_fixed_demonstrations.json')==protocol['fixed25_sha256'];prefix=joint_prefix(read('data/slcd_fixed_demonstrations.json'))
    r10=read('configs/r10_PROTOCOL.json');assert __import__('hashlib').sha256(prefix.encode()).hexdigest()==r10['joint_demonstrations_sha256']
    if a.phase=='profile':
        result=profile(model,scorer,tok,data,names,prefix,lock)
        after=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before==after
        gate(a.phase)
        print('R11_PROFILE_COMPLETE',json.dumps(result),flush=True);return
    robust='robust_cyclic_loop_axis' in names;historical={};dev={};maxerror=0.;maxpeak=0
    if a.phase=='parity':
        d=read('results/r11_DEVELOPMENT_COMPLETE.json');assert sha(d['score_file'])==d['score_file_sha256']
        rr=read('results/r10_development_initial_COMPLETE.json');assert sha('results/r10_development_initial_COMPLETE.json')==d['r10_auxiliary_receipt_sha256'] and sha(rr['score_file'])==rr['score_file_sha256']
        historical={r['id']:r for r in jsonlines(rr['score_file'])};dev={r['id']:r for r in jsonlines(d['score_file'])}
    for value in data:
        row=value['item']
        if row['id'] in done:continue
        check_budget(30);begin=time.time();torch.cuda.reset_peak_memory_stats();clean=dict(question=row['question'],choices=row['choices'])
        aux=infer_auxiliary(model,scorer,tok,clean,prefix,robust);scores,diags=select(value['original'],value['prior'],aux,names);joint=joint_baselines(aux)
        if a.phase=='parity':
            old=historical[row['id']];n=len(clean['choices'])
            assert aux['target_ids_sha256'][:n]==old['target_ids_sha256'] and aux['target_token_counts']==old['target_token_counts']
            assert len(aux['cyclic_rotations'])==len(old['cyclic_rotations'])
            assert aux['cyclic_rotations'][0]['request_sha256']==old['auxiliary_request_sha256']
            for index,(nr,hr) in enumerate(zip(aux['cyclic_rotations'],old['cyclic_rotations'])):
                assert nr['order']==hr['order']
                if index:assert nr['request_sha256']==hr['request_sha256']
                for k,v in nr['sum_scores'].items():maxerror=max(maxerror,float(np.max(np.abs(np.asarray(v)-np.asarray(hr['sum_scores'][k])))))
            for k,v in scores.items():maxerror=max(maxerror,float(np.max(np.abs(v-np.asarray(dev[row['id']]['method_scores'][k])))))
            assert maxerror==0.,('Fresh K4 vs historical K5 readouts or frozen scores',row['id'],maxerror)
        vectors={**value['baseline_scores'],**{k:v.tolist() for k,v in scores.items()},**joint};pred={k:int(np.argmax(v)) for k,v in vectors.items()};gold=row['gold']
        torch.cuda.synchronize();peak=int(torch.cuda.max_memory_allocated());maxpeak=max(maxpeak,peak)
        record=dict(id=row['id'],split=value['split'],signature=sig,prompt_sha256=row['prompt_sha256'],gold=gold,baseline_scores=value['baseline_scores'],
            method_scores={k:v.tolist() for k,v in scores.items()},joint_baseline_scores=joint,diagnostics=diags,**aux,
            predictions=pred,correct={k:v==gold for k,v in pred.items()},elapsed_seconds=time.time()-begin,peak_cuda_allocated_bytes=peak)
        with (ROOT/output).open('a',encoding='utf8') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        done[row['id']]=record;progress=dict(phase=a.phase,completed=len(done),target=len(data),elapsed_seconds=time.time()-started,
            correct={k:sum(r['correct'][k] for r in done.values()) for k in vectors});save('results/r11_'+a.phase+'_PROGRESS.json',progress)
        if len(done)%50==0 or len(done)==len(data):print(json.dumps(progress),flush=True)
    after=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before==after;gate(a.phase)
    records=[done[v['item']['id']] for v in data]
    result=dict(status='complete',phase=a.phase,n=len(records),names=names,signature=sig,score_file=output,score_file_sha256=sha(output),
        protocol_sha256=sha('configs/r11_PROTOCOL.json'),definition_lock_sha256=sha('configs/r11_CANDIDATE_DEFINITION_LOCK.json'),stage_lock_sha256=sha(lockpath),
        extra_source_sha256=extra_sources(),inference_definition_sha256=protocol['source_sha256'],head_norm_before=before,head_norm_after=after,
        weights_modified=False,max_parity_error=maxerror if a.phase=='parity' else None,elapsed_seconds=time.time()-started,
        actual_auxiliary_depth=4,original_main_prior_scores_reused=True,KV_cache=False,attention_mask_argument=False,peak_cuda_allocated_bytes=maxpeak,
        parity_historical_target_hash_scope='Original-order n candidates only; cyclic hashes were not stored in the historical authority. Full cyclic request hashes, target counts and LLs are independently compared.' if a.phase=='parity' else None)
    save(receipt,result);print('R11_SELECTED_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
