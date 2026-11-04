"""Frozen, label-free R7 scorer. Original uncached attention path only.

Model head operates on each complete continuation before vocabulary statistics
are streamed in blocks of32 prediction positions. Gold is never accepted by
the numerical scoring functions. Stored diagnostic arrays have no vocab axis.
"""
import time
import numpy as np
from r7_math import candidate_scores,token_logits

BLOCK=32
TOKEN_MODES={'M2_N0_token':('robust',1),'M3_multi_token':('robust',3),
 'C_mean_denominator':('mean_only',3),'C_R6alpha_token':('bridge',1),
 'C_matched_guidance':('matched_scale',3),'C_shuffled_direction':('shuffle',3)}


def forward_states(scorer,pair,depth):
    scorer.kmax=depth;scorer._set_depth(depth)
    prefix,target=pair
    _,states,_=scorer._run(prefix+target[:-1],positions=range(len(prefix)-1,len(prefix)+len(target)-1),use_cache=False)
    return states


def head_logits(model,states):
    import torch
    values={}
    for k in (1,4,5):
        if k<=len(states):
            h=states[k-1].to(model.lm_head.weight.device)
            assert h.dtype==torch.bfloat16 and h.ndim==3 and h.shape[0]==1
            z=model.lm_head(h)[0];assert z.dtype==torch.bfloat16
            values[k]=z.float()
    return values


def target_lp(logits,targets):
    import torch
    t=torch.as_tensor(targets,device=logits.device,dtype=torch.long)
    result=logits.log_softmax(-1).gather(-1,t[:,None]).squeeze(-1)
    assert bool(torch.isfinite(result).all())
    return result


def original_sums(logits,target):
    z1,z4=logits[1],logits[4]
    result={f'native_k{k}':float(target_lp(z,target).sum()) for k,z in logits.items()}
    fixed=z4+.5*(z4-z1);result['loopcd_k4_fixed']=float(target_lp(fixed,target).sum())
    if 5 in logits:
        top=z4.softmax(-1).topk(2,dim=-1).values
        adaptive=z4+(1-(top[:,0]-top[:,1]))[:,None]*(z4-z1)
        result['loopcd_k4_adaptive']=float(target_lp(adaptive,target).sum())
    return result


def methods_from_sums(sums,lengths):
    return {k:dict(sum_scores=v,normalized_scores=(np.asarray(v)/lengths).tolist()) for k,v in sums.items()}


def score_token_candidate(q,refs,maps,target,names,whole_alpha,permutation,checks=False):
    import torch
    assert len(q[1])==len(target)
    wanted=set(names);active=wanted&set(TOKEN_MODES)
    if 'C_no_late' in wanted:active.add('M3_multi_token')
    pieces={name:[] for name in active};diags={name:[] for name in active}
    real_checks={}
    for begin in range(0,len(target),BLOCK):
        end=min(begin+BLOCK,len(target));q1=q[1][begin:end];q4=q[4][begin:end]
        reference1=torch.stack([r[1][torch.as_tensor(m[begin:end],device=q1.device)] for r,m in zip(refs,maps)])
        reference4=torch.stack([r[4][torch.as_tensor(m[begin:end],device=q1.device)] for r,m in zip(refs,maps)])
        fixed=q4+.5*(q4-q1);fixed_lp=target_lp(fixed,target[begin:end]);t=torch.as_tensor(target[begin:end],device=q1.device)
        for name in sorted(active):
            mode,j=TOKEN_MODES[name]
            kwargs={'whole_alpha':whole_alpha} if mode=='bridge' else {'permutation':permutation} if mode=='shuffle' else {}
            z,d=token_logits(q1,q4,reference1[:j],reference4[:j],mode=mode,**kwargs)
            lp=target_lp(z,target[begin:end]);pieces[name].append(lp)
            # Full normalizer's contribution is distinct from target-logit shift.
            target_shift=(z-fixed).gather(-1,t[:,None]).squeeze(-1)
            packed={k:v.detach().cpu().tolist() for k,v in d.items() if isinstance(v,torch.Tensor)}
            packed.update(target_logit_shift=target_shift.cpu().tolist(),
                          target_logprob_change=(lp-fixed_lp).cpu().tolist(),
                          log_normalizer_change=(target_shift-(lp-fixed_lp)).cpu().tolist())
            diags[name].append(packed)
        if checks and begin==0:
            zero,_=token_logits(q1,q4,reference1,reference4,force_zero=True)
            assert torch.equal(zero,fixed)
            one,_=token_logits(q1,q4,reference1[:1],reference4[:1])
            duplicate,_=token_logits(q1,q4,reference1[:1].expand(3,-1,-1),reference4[:1].expand(3,-1,-1))
            assert torch.equal(one,duplicate)
            # Candidate extension cannot affect this position-only API; use an
            # independent re-call and compare preserved prefix outputs exactly.
            extension,_=token_logits(torch.cat([q1,q1[:1]]),torch.cat([q4,q4[:1]]),
                torch.cat([reference1,reference1[:,:1]],dim=1),torch.cat([reference4,reference4[:,:1]],dim=1))
            original,_=token_logits(q1,q4,reference1,reference4)
            assert torch.equal(extension[:-1],original)
            real_checks.update(zero_exact_fixed=True,identical_references_exact=True,candidate_extension_exact=True)
    # Concatenate token likelihoods FIRST; native FP32 sum order is preserved.
    sums={name:float(torch.cat(parts).sum()) for name,parts in pieces.items()}
    if 'C_no_late' in wanted:sums['C_no_late']=sums['M3_multi_token']
    sums={name:v for name,v in sums.items() if name in wanted}
    merged={name:{k:sum((block[k] for block in blocks),[]) for k in blocks[0]} for name,blocks in diags.items()}
    return sums,merged,real_checks


def infer_question(model,scorer,tok,row,names,reference_prompts,main_states=None,checks=False):
    """row requires request_pairs and lengths; this function never reads gold."""
    import torch
    from runtime import encode_pair
    from r7_references import prefix_alignment
    from slcd_r6_scores import _inputs,projection_parameters
    pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
    counts={name:(1 if name in ['M2_N0_token','C_R6alpha_token'] else 3) for name in names}
    required=max(counts.values())
    refpairs=[[encode_pair(tok,prompt,request[1]) for request in row['request_pairs']] for prompt in reference_prompts[:required]]
    alignments={j:[prefix_alignment(tok,request,pair,[rp[i] for rp in refpairs[:j]]) for i,(request,pair) in enumerate(zip(row['request_pairs'],pairs))] for j in set(counts.values())}
    fallback_by_method={name:any(m is None for m,reason in alignments[j]) for name,j in counts.items()}
    active_names=[name for name in names if not fallback_by_method[name]]
    forward_references=max([1]+[counts[name] for name in active_names])
    lengths=np.asarray(row['choice_char_lengths'],dtype=np.float64)
    main_sums={};prior_sums=[{}, {}, {}];qlogits=[];rlogits=[]
    timing=dict(main_forward=0.,reference_forward=[0.,0.,0.],head_and_projection=0.)
    for i,pair in enumerate(pairs):
        torch.cuda.synchronize();start=time.time()
        states=forward_states(scorer,pair,5) if main_states is None else main_states[i]['normalized']
        if main_states is not None:assert main_states[i]['target_ids']==pair[1] and main_states[i]['candidate']==i
        torch.cuda.synchronize();timing['main_forward']+=time.time()-start;start=time.time()
        q=head_logits(model,states);qlogits.append(q)
        for k,v in original_sums(q,pair[1]).items():main_sums.setdefault(k,[]).append(v)
        timing['head_and_projection']+=time.time()-start
        rr=[]
        for j in range(forward_references):
            torch.cuda.synchronize();start=time.time();rs=forward_states(scorer,refpairs[j][i],4)
            torch.cuda.synchronize();timing['reference_forward'][j]+=time.time()-start;start=time.time()
            rz=head_logits(model,rs);rr.append(rz)
            for k,v in original_sums(rz,refpairs[j][i][1]).items():prior_sums[j].setdefault(k,[]).append(v)
            timing['head_and_projection']+=time.time()-start
        rlogits.append(rr)
    original=methods_from_sums(main_sums,lengths)
    prior=[methods_from_sums(s,lengths) for s in prior_sums if s]
    from r7_core import baseline_vectors
    baselines=baseline_vectors(original,prior[0]);a,g,e=_inputs(original,prior[0]);alpha,_,_,_=projection_parameters(g,e)
    one,d1=candidate_scores(original,prior[:1]);assert np.array_equal(one,baselines['M0_R6'])
    methods={name:baselines['M0_R6'].copy() for name in names if fallback_by_method[name]};m1diag=None;details=[];real_checks={}
    if active_names:
        if 'M1_multi_candidate' in active_names:methods['M1_multi_candidate'],m1diag=candidate_scores(original,prior)
        sums={name:[] for name in active_names if name!='M1_multi_candidate'}
        permutation=torch.as_tensor(np.random.default_rng(2021).permutation(model.config.vocab_size),device=scorer.device)
        for i,(_,target) in enumerate(pairs):
            torch.cuda.synchronize();start=time.time()
            mappings=alignments[forward_references][i][0]
            values,diag,check=score_token_candidate(qlogits[i],rlogits[i],mappings,target,active_names,alpha,permutation,checks=checks and i==0)
            torch.cuda.synchronize();timing['head_and_projection']+=time.time()-start
            for name,value in values.items():sums[name].append(value)
            details.append(diag);real_checks.update(check)
        late=baselines['native4']-np.asarray(original['native_k5']['normalized_scores'])
        for name,values in sums.items():methods[name]=np.asarray(values)/lengths+(0. if name=='C_no_late' else late)
    assert set(methods)==set(names) and all(np.isfinite(v).all() for v in methods.values())
    return dict(method_scores={k:v.tolist() for k,v in methods.items()},baseline_scores={k:v.tolist() for k,v in baselines.items()},
        original_methods=original,prior_methods=prior,alpha6=alpha,M1_diagnostic=m1diag,token_diagnostics=details,
        alignment_fallback=any(fallback_by_method.values()),alignment_fallback_by_method=fallback_by_method,
        alignment_reasons={str(j):[reason for m,reason in values] for j,values in alignments.items()},
        aligned_candidates={str(j):sum(m is not None for m,reason in values) for j,values in alignments.items()},candidate_count=len(pairs),
        references_forwarded=forward_references,
        token_counts=[len(t) for p,t in pairs],timing=timing,real_checks=real_checks,
        gold_free_function_contract='Only request_pairs and choice_char_lengths accessed; no gold/correctness input')
