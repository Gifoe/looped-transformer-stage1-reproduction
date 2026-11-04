"""Actual5/4 deployment timing, no hooks, no historical states or KV reuse.

This mirrors the frozen mathematical operations and checks every scored vector
against the completed discovery artifact. No diagnostic arrays are copied to
CPU; no accuracy comparison selects a method here.
"""
import time
import numpy as np
from r7_core import ROOT,read,save,sha,gated_rows,jsonlines


def main():
    import torch
    from runtime import load_model,encode_pair,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from r7_inference import forward_states,head_logits,original_sums,methods_from_sums,target_lp,BLOCK
    from r7_references import prefix_alignment
    from r7_math import candidate_scores,token_logits
    from r7_core import baseline_vectors
    lock=read('configs/r7_FINAL_METHOD_LOCK.json');primary=lock['primary']
    for n,value in lock['inference_source_sha256'].items():assert sha('scripts/'+n)==value
    receipt=read('results/r7_discovery_development_COMPLETE.json')
    assert receipt['status']=='complete' and sha('results/r7_discovery_development.jsonl')==receipt['score_file_sha256']
    saved={r['id']:r for r in jsonlines('results/r7_discovery_development.jsonl')};assert len(saved)==receipt['n']==384
    assert all(r['signature']==receipt['signature'] for r in saved.values())
    data=gated_rows('discovery')[:12];start=time.time();tok,model=load_model();scorer=OuroCachedScorer(model,5)
    torch.cuda.synchronize();load_seconds=time.time()-start
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    j=1 if primary=='M2_N0_token' else 3
    refmanifest=read('configs/r7_reference_manifest.json')['references'];prompts=[refmanifest[k] for k in ['N0','N1','N2']][:j]
    records=[]
    for row in data:
        check_budget(45);torch.cuda.synchronize();begin=time.time();torch.cuda.reset_peak_memory_stats()
        pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
        refpairs=[[encode_pair(tok,prompt,request[1]) for request in row['request_pairs']] for prompt in prompts]
        alignment=[prefix_alignment(tok,request,pair,[rp[i] for rp in refpairs]) for i,(request,pair) in enumerate(zip(row['request_pairs'],pairs))]
        failed=any(m is None for m,reason in alignment);orig={};ps=[{} for _ in range(j)];totals=[]
        timing=dict(main=0.,references=[0.]*j,head=0.,projection=0.)
        for i,pair in enumerate(pairs):
            torch.cuda.synchronize();t=time.time();states=forward_states(scorer,pair,5);torch.cuda.synchronize();timing['main']+=time.time()-t
            t=time.time();q=head_logits(model,states)
            for name,value in original_sums(q,pair[1]).items():orig.setdefault(name,[]).append(value)
            torch.cuda.synchronize();timing['head']+=time.time()-t;rs=[]
            for k in range(j):
                t=time.time();states=forward_states(scorer,refpairs[k][i],4);torch.cuda.synchronize();timing['references'][k]+=time.time()-t
                t=time.time();r=head_logits(model,states);rs.append(r)
                for name,value in original_sums(r,refpairs[k][i][1]).items():ps[k].setdefault(name,[]).append(value)
                torch.cuda.synchronize();timing['head']+=time.time()-t
            if primary!='M1_multi_candidate' and not failed:
                t=time.time();parts=[]
                for b in range(0,len(pair[1]),BLOCK):
                    e=min(b+BLOCK,len(pair[1]));a=alignment[i][0]
                    r1=torch.stack([r[1][a[k][b:e]] for k,r in enumerate(rs)])
                    r4=torch.stack([r[4][a[k][b:e]] for k,r in enumerate(rs)])
                    z,_=token_logits(q[1][b:e],q[4][b:e],r1,r4)
                    parts.append(target_lp(z,pair[1][b:e]))
                totals.append(float(torch.cat(parts).sum()));torch.cuda.synchronize();timing['projection']+=time.time()-t
        lengths=np.asarray(row['choice_char_lengths']);original=methods_from_sums(orig,lengths);priors=[methods_from_sums(p,lengths) for p in ps]
        base=baseline_vectors(original,priors[0])
        if failed:score=base['M0_R6']
        elif primary=='M1_multi_candidate':score,_=candidate_scores(original,priors)
        else:score=np.asarray(totals)/lengths+(base['native4']-np.asarray(original['native_k5']['normalized_scores']))
        error=float(np.max(np.abs(score-np.asarray(saved[row['id']]['method_scores'][primary]))));assert error==0.,('Deployment parity',row['id'],error)
        torch.cuda.synchronize();records.append(dict(id=row['id'],seconds=time.time()-begin,timing=timing,
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),score_max_error=error,alignment_fallback=failed))
    # Same fixed12 questions, independent original5/4 R6 execution. This is
    # timing only; no score here can alter the frozen primary or test gate.
    def r6_sample(row):
        original={};prior={}
        for request in row['request_pairs']:
            pair=encode_pair(tok,*request);rp=encode_pair(tok,'Answer:',request[1])
            q=head_logits(model,forward_states(scorer,pair,5));p=head_logits(model,forward_states(scorer,rp,4))
            for key,value in original_sums(q,pair[1]).items():original.setdefault(key,[]).append(value)
            for key,value in original_sums(p,rp[1]).items():prior.setdefault(key,[]).append(value)
        lengths=np.asarray(row['choice_char_lengths'])
        return baseline_vectors(methods_from_sums(original,lengths),methods_from_sums(prior,lengths))['M0_R6']
    r6_records=[]
    for row in data:
        check_budget(30);torch.cuda.synchronize();begin=time.time();score=r6_sample(row);torch.cuda.synchronize()
        assert np.array_equal(score,np.asarray(saved[row['id']]['baseline_scores']['M0_R6']))
        r6_records.append(dict(id=row['id'],seconds=time.time()-begin,score_max_error=0.))
    r6_warm=float(np.mean([r['seconds'] for r in r6_records[1:]]))
    # Required real-model invariants are separate from all timed samples.
    # The first fixed discovery question includes a34-token continuation and
    # therefore exercises accumulation across the32-position block boundary.
    invariant_row=data[0];zero_errors=[];shift_errors=[];max_tokens=0
    for request,length in zip(invariant_row['request_pairs'],invariant_row['choice_char_lengths']):
        check_budget(30);pair=encode_pair(tok,*request);rp=encode_pair(tok,'Answer:',request[1])
        mapping,reason=prefix_alignment(tok,request,pair,[rp]);assert mapping is not None,reason
        q=head_logits(model,forward_states(scorer,pair,5));r=head_logits(model,forward_states(scorer,rp,4))
        r1=r[1][mapping[0]][None];r4=r[4][mapping[0]][None];parts=[]
        for b in range(0,len(pair[1]),BLOCK):
            e=min(b+BLOCK,len(pair[1]));zero,_=token_logits(q[1][b:e],q[4][b:e],r1[:,b:e],r4[:,b:e],force_zero=True)
            assert torch.equal(zero,q[4][b:e]+.5*(q[4][b:e]-q[1][b:e]))
            parts.append(target_lp(zero,pair[1][b:e]))
        new=float(torch.cat(parts).sum());native=original_sums(q,pair[1]);error=abs(new-native['loopcd_k4_fixed']);assert error==0.
        late=native['native_k4']/length-native['native_k5']/length
        assert new/length+late==native['loopcd_k4_fixed']/length+late
        zero_errors.append(error);max_tokens=max(max_tokens,len(pair[1]))
        base,d=token_logits(q[1][:1],q[4][:1],r1[:,:1],r4[:,:1])
        shifted,sd=token_logits(q[1][:1]+.25,q[4][:1]+.25,r1[:,:1]-.5,r4[:,:1]-.5)
        aerr=float((d['alpha']-sd['alpha']).abs().max());perr=float((base.log_softmax(-1)-shifted.log_softmax(-1)).abs().max())
        assert aerr<=3e-5 and perr<=3e-5,('FP32 constant-shift invariance',aerr,perr)
        shift_errors.append(dict(max_alpha_abs_error=aerr,max_full_vocabulary_logprob_abs_error=perr))
    assert max_tokens>BLOCK
    real_invariants=dict(status='passed',question=invariant_row['id'],candidates=len(zero_errors),max_continuation_tokens=max_tokens,
        block_size=BLOCK,zero_correction_complete_sum_max_abs_error=max(zero_errors),zero_correction_recovers_A1=True,
        constant_shift_tolerance=3e-5,constant_shift_max_alpha_abs_error=max(r['max_alpha_abs_error'] for r in shift_errors),
        constant_shift_max_full_vocabulary_logprob_abs_error=max(r['max_full_vocabulary_logprob_abs_error'] for r in shift_errors),
        scope='Independent original model5/4; no diagnostic checks included in timed records')
    import resource
    save('results/r7_DEPLOYMENT_PROFILE.json',dict(status='complete',n=len(data),primary=primary,
        source_sha256=sha('scripts/r7_deploy.py'),final_method_lock_sha256=sha('configs/r7_FINAL_METHOD_LOCK.json'),
        actual_main_depth=5,actual_reference_depth=4,reference_count=j,diagnostic_hooks=False,diagnostic_CPU_array_export=False,
        historical_main_states=False,KV_cache=False,reference_prefix_cache=False,reference_cache_warmup_seconds=None,
        load_seconds=load_seconds,first_question_seconds=records[0]['seconds'],
        warm_mean_question_seconds=float(np.mean([r['seconds'] for r in records[1:]])),
        warm_median_question_seconds=float(np.median([r['seconds'] for r in records[1:]])),
        R6_warm_mean_question_seconds=r6_warm,R6_records=r6_records,
        warm_time_ratio_vs_R6=float(np.mean([r['seconds'] for r in records[1:]]))/r6_warm,
        timing_design='Sequential blocks on identical12 questions, first question excluded from both warm means; local deployment timing, not a hardware-wide speed guarantee',
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in records),
        host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_semantics='PyTorch allocator bytes on GB10 unified memory, not a discrete-VRAM claim',records=records,
        real_model_invariants=real_invariants))
    print('R7_DEPLOYMENT_COMPLETE',flush=True)


if __name__=='__main__':main()
