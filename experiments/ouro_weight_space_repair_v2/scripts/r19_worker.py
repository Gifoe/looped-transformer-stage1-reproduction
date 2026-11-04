"""Actual loop3 information under the unchanged prompt, no label in scoring."""
import json,os,time
import numpy as np
from r19_common import ROOT,LOCK,read,sha,digest,save,gate,inputs


def third_head(model,states):
    import torch
    h=states[2].to(model.lm_head.weight.device);assert h.dtype==torch.bfloat16 and h.ndim==3 and h.shape[0]==1
    z=model.lm_head(h)[0];assert z.dtype==torch.bfloat16;return z.float()


def measure(model,scorer,tok,request,hidden,ref,expected,index,characters,permutation,check=False):
    import torch
    from runtime import encode_pair,check_budget
    from r7_inference import head_logits,forward_states,original_sums,target_lp
    from r7_references import prefix_alignment
    from r7_math import token_logits
    from r19_math import torch_logits,numpy_primary
    start=time.monotonic();pair=encode_pair(tok,*request);assert hidden['candidate']==index and hidden['target_ids']==pair[1]
    q=head_logits(model,hidden['normalized']);original=original_sums(q,pair[1])
    for k,value in original.items():assert value==expected['original_methods'][k]['sum_scores'][index],('Main parity',index,k)
    q[3]=third_head(model,hidden['normalized']);q3sum=float(target_lp(q[3],pair[1]).sum());assert q3sum==expected['native3_sum_scores'][index],('Original native3 parity',index,q3sum)
    rp=encode_pair(tok,ref,request[1]);maps,reason=prefix_alignment(tok,request,pair,[rp]);assert maps is not None,('Unsupported prefix',reason)
    torch.cuda.synchronize();head_seconds=time.monotonic()-start;start=time.monotonic();check_budget(5)
    refstates=forward_states(scorer,rp,4);r=head_logits(model,refstates);observed=original_sums(r,rp[1])
    for k,value in observed.items():assert value==expected['prior_methods'][0][k]['sum_scores'][index],('N0 parity',index,k)
    r[3]=third_head(model,refstates);depth_check=None
    if check:
        freshQ3=third_head(model,forward_states(scorer,pair,3));freshN3=third_head(model,forward_states(scorer,rp,3))
        assert torch.equal(freshQ3,q[3]) and torch.equal(freshN3,r[3]),'Independent depth3 full-vocabulary head parity'
        depth_check=dict(main_cached6_vs_independent3_full_vocab_exact=True,reference_shared4_vs_independent3_full_vocab_exact=True)
    torch.cuda.synchronize();reference_seconds=time.monotonic()-start;start=time.monotonic()
    names=['paired_loop_protection','reference_only_axis','matched_guidance','shuffled_correction','shuffled_axis','old_Fisher'];parts={k:[] for k in names};packed={};cpu_check=None
    for begin in range(0,len(pair[1]),32):
        end=min(begin+32,len(pair[1]));ix=torch.as_tensor(maps[0][begin:end],device=q[4].device)
        values=[q[1][begin:end],q[3][begin:end],q[4][begin:end],r[1][ix].unsqueeze(0),r[3][ix].unsqueeze(0),r[4][ix].unsqueeze(0)]
        controls,diag=torch_logits(*values,permutation=permutation)
        old,od=token_logits(values[0],values[2],values[3],values[5]);assert torch.equal(od['alpha'],diag['old_Fisher_alpha']);controls['old_Fisher']=old
        fixed=values[2]+.5*(values[2]-values[0]);base=target_lp(fixed,pair[1][begin:end]);targets=torch.as_tensor(pair[1][begin:end],device=fixed.device);arrays={k:v.cpu().tolist() for k,v in diag.items()}
        for k,z in controls.items():
            lp=target_lp(z,pair[1][begin:end]);parts[k].append(lp);shift=(z-fixed).gather(-1,targets[:,None]).squeeze(-1)
            arrays[k+'_LP']=lp.cpu().tolist();arrays[k+'_target_shift']=shift.cpu().tolist();arrays[k+'_LP_change']=(lp-base).cpu().tolist();arrays[k+'_normalizer_change']=(shift-(lp-base)).cpu().tolist()
        if check and begin==0:
            subset=[x[:1] if x.ndim==2 else x[:,:1] for x in values]
            zz,_=torch_logits(*subset,permutation=permutation,force_zero=True);assert all(torch.equal(v,fixed[:1]) for v in zz.values())
            independent,nd=numpy_primary(*(x.cpu().numpy() for x in subset));ae=float(np.max(np.abs(nd['alpha']-diag['alpha'][:1].cpu().numpy())));ze=float(np.max(np.abs(independent-controls['paired_loop_protection'][:1].cpu().numpy())))
            assert ae<=2**-32 and ze<=2e-5;cpu_check=dict(alpha_error=ae,FP32_logit_error=ze,zero_exact_Fixed=True)
        for k,v in arrays.items():packed.setdefault(k,[]).extend(v)
    sums={k:float(torch.cat(v).sum()) for k,v in parts.items()};length=int(characters);assert length>0
    late=expected['original_methods']['native_k4']['normalized_scores'][index]-expected['original_methods']['native_k5']['normalized_scores'][index]
    scores={k:v/length+late for k,v in sums.items()};assert scores['old_Fisher']==expected['M2_scores'][index] and packed['old_Fisher_alpha']==expected['token_diagnostics'][index]['M2_N0_token']['alpha']
    assert all(len(v)==len(pair[1]) for v in packed.values());torch.cuda.synchronize()
    return dict(choice=index,characters=length,tokens=len(pair[1]),sums=sums,scores=scores,late=late,metrics=packed,parity_error=0.,original_sums=original,original_native3_sum=q3sum,
        reference_sums=observed,CPU_numpy_check=cpu_check,independent_depth3_check=depth_check,request_sha256=digest(request),target_ids_sha256=digest(pair[1]),reference_positions=maps[0],
        timing=dict(main_head_seconds=head_seconds,N0_reference_and_independent3_seconds=reference_seconds,projection_and_controls_seconds=time.monotonic()-start))


def main():
    started=time.monotonic();spec=gate();data=inputs(spec)
    assert not any((ROOT/p).exists() for p in ['results/r19_pilot.jsonl','results/r19_pilot_COMPLETE.json']),'No automatic retry/resume'
    from r19_math import synthetic_checks
    save('results/r19_MATH_CHECKS.json',synthetic_checks())
    from runtime import load_model,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from weight_edits import tensor_sha256
    import torch
    t=time.monotonic();tok,model=load_model();scorer=OuroCachedScorer(model,5);load_seconds=time.monotonic()-t;identity=spec['model_identity']
    assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for p,h in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+p)==h
    before=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before['head']==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    permutation=np.random.default_rng(2021).permutation(model.config.vocab_size);ref=read('configs/r7_reference_manifest.json')['references']['N0'];signature=digest(dict(protocol_sha256=sha(LOCK),cohort=spec['cohort']));count=0;peak=0
    with (ROOT/'results/r19_pilot.jsonl').open('x',encoding='utf8') as out:
        for chosen,clean,expected,state in data:
            check_budget(5);payload=torch.load(ROOT/state['file'],map_location='cpu',weights_only=True);assert payload['id']==chosen['id'];torch.cuda.reset_peak_memory_stats()
            candidates=[measure(model,scorer,tok,request,payload['candidate_states'][j],ref,expected,j,clean['choice_char_lengths'][j],permutation,check=count==0 and j==0) for j,request in enumerate(clean['request_pairs'])]
            row=dict(id=chosen['id'],split=chosen['split'],signature=signature,cohort_entry_sha256=digest(chosen),state_file_sha256=state['sha256'],candidates=candidates)
            out.write(json.dumps(row,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno());count+=1;peak=max(peak,torch.cuda.max_memory_allocated());save('results/r19_pilot_PROGRESS.json',dict(completed=count,target=64,elapsed_seconds=time.monotonic()-started))
    after=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before==after and count==64
    save('results/r19_pilot_COMPLETE.json',dict(status='complete',n=count,protocol_sha256=sha(LOCK),signature=signature,score_file='results/r19_pilot.jsonl',score_file_sha256=sha('results/r19_pilot.jsonl'),
        head_norm_before=before,head_norm_after=after,weights_modified=False,peak_cuda_allocated_bytes=peak,elapsed_seconds=time.monotonic()-started,model_load_seconds=load_seconds,
        cached_main_states=True,new_main_forwards='One independent depth3 parity candidate only',new_reference_depth=4,KV_cache=False,explicit_attention_mask=False,heldout_accessed=False,standalone_profile=False))
    print('R19_FULL_PILOT_COMPLETE',count,flush=True)


if __name__=='__main__':main()
