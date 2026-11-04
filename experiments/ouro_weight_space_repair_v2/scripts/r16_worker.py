"""Actual full-vocabulary J1 native-KL calibration; no gold enters scoring."""
import json,os,time
import numpy as np
from r16_common import ROOT,LOCK,read,sha,digest,save,gate,inputs


def measure(model,scorer,tok,request,hidden,ref,expected,index,characters,permutation,check=False):
    import torch
    from runtime import encode_pair,check_budget
    from r7_inference import head_logits,forward_states,original_sums,target_lp
    from r7_references import prefix_alignment
    from r7_math import token_logits
    from r16_math import torch_logits,numpy_logits
    start=time.monotonic();pair=encode_pair(tok,*request)
    assert hidden['candidate']==index and hidden['target_ids']==pair[1]
    q=head_logits(model,hidden['normalized']);original=original_sums(q,pair[1])
    for k,v in original.items():assert v==expected['original_methods'][k]['sum_scores'][index],('Main parity',index,k)
    rp=encode_pair(tok,ref,request[1]);maps,reason=prefix_alignment(tok,request,pair,[rp]);assert maps is not None,('Unsupported prefix',reason)
    torch.cuda.synchronize();head_seconds=time.monotonic()-start;start=time.monotonic();check_budget(3)
    r=head_logits(model,forward_states(scorer,rp,4));observed=original_sums(r,rp[1])
    for k,v in observed.items():assert v==expected['prior_methods'][0][k]['sum_scores'][index],('N0 parity',index,k)
    torch.cuda.synchronize();reference_seconds=time.monotonic()-start;start=time.monotonic()
    parts={k:[] for k in ['exact_native_KL','matched_guidance','shuffled_correction','old_Fisher']};packed={};cpu_check=None
    for begin in range(0,len(pair[1]),32):
        end=min(begin+32,len(pair[1]));q1,q4=q[1][begin:end],q[4][begin:end];ix=torch.as_tensor(maps[0][begin:end],device=q4.device)
        r1,r4=r[1][ix].unsqueeze(0),r[4][ix].unsqueeze(0)
        controls,diag=torch_logits(q1,q4,r1,r4,permutation=permutation)
        old,od=token_logits(q1,q4,r1,r4);assert torch.equal(od['alpha'],diag['old_Fisher_alpha'])
        controls['old_Fisher']=old
        fixed=q4+.5*(q4-q1);base=target_lp(fixed,pair[1][begin:end]);targets=torch.as_tensor(pair[1][begin:end],device=q4.device)
        arrays={k:v.cpu().tolist() for k,v in diag.items()}
        for k,z in controls.items():
            lp=target_lp(z,pair[1][begin:end]);parts[k].append(lp)
            shift=(z-fixed).gather(-1,targets[:,None]).squeeze(-1)
            arrays[k+'_LP']=lp.cpu().tolist();arrays[k+'_target_shift']=shift.cpu().tolist()
            arrays[k+'_LP_change']=(lp-base).cpu().tolist();arrays[k+'_normalizer_change']=(shift-(lp-base)).cpu().tolist()
        if check and begin==0:
            zz,zd=torch_logits(q1[:1],q4[:1],r1[:,:1],r4[:,:1],permutation=permutation,force_zero=True)
            assert all(torch.equal(z,fixed[:1]) for z in zz.values())
            cpu,cd=numpy_logits(*(x[:1].cpu().numpy() if x.ndim==2 else x[:,:1].cpu().numpy() for x in [q1,q4,r1,r4]))
            ae=float(np.max(np.abs(cd['alpha']-diag['alpha'][:1].cpu().numpy())));ze=float(np.max(np.abs(cpu-controls['exact_native_KL'][:1].cpu().numpy())))
            assert ae<=2**-32 and ze<=2e-5
            cpu_check=dict(alpha_error=ae,FP32_logit_error=ze,zero_exact_Fixed=True)
        for k,v in arrays.items():packed.setdefault(k,[]).extend(v)
    sums={k:float(torch.cat(v).sum()) for k,v in parts.items()};length=int(characters);assert length>0
    late=expected['original_methods']['native_k4']['normalized_scores'][index]-expected['original_methods']['native_k5']['normalized_scores'][index]
    scores={k:v/length+late for k,v in sums.items()}
    assert scores['old_Fisher']==expected['M2_scores'][index]
    assert packed['old_Fisher_alpha']==expected['token_diagnostics'][index]['M2_N0_token']['alpha']
    assert all(len(v)==len(pair[1]) for v in packed.values())
    torch.cuda.synchronize();solver_seconds=time.monotonic()-start
    return dict(choice=index,characters=length,tokens=len(pair[1]),sums=sums,scores=scores,late=late,metrics=packed,
        parity_error=0.,original_sums=original,reference_sums=observed,CPU_numpy_check=cpu_check,request_sha256=digest(request),
        target_ids_sha256=digest(pair[1]),reference_positions=maps[0],timing=dict(main_head_seconds=head_seconds,N0_reference_seconds=reference_seconds,solver_and_controls_seconds=solver_seconds))


def main():
    started=time.monotonic();spec=gate();data=inputs(spec)
    assert not any((ROOT/p).exists() for p in ['results/r16_pilot.jsonl','results/r16_pilot_COMPLETE.json']),'No automatic retry/resume'
    from r16_math_checks import main as checks
    save('results/r16_MATH_CHECKS.json',checks())
    from runtime import load_model,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from weight_edits import tensor_sha256
    import torch
    before_load=time.monotonic();tok,model=load_model();scorer=OuroCachedScorer(model,5);load_seconds=time.monotonic()-before_load
    identity=spec['model_identity'];assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for p,h in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+p)==h
    before=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before['head']==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    permutation=np.random.default_rng(2021).permutation(model.config.vocab_size);ref=read('configs/r7_reference_manifest.json')['references']['N0']
    signature=digest(dict(protocol_sha256=sha(LOCK),cohort=spec['cohort']));count=0;peak=0
    with (ROOT/'results/r16_pilot.jsonl').open('x',encoding='utf8') as out:
        for chosen,clean,expected,state in data:
            check_budget(5);payload=torch.load(ROOT/state['file'],map_location='cpu',weights_only=True);assert payload['id']==chosen['id']
            torch.cuda.reset_peak_memory_stats()
            candidates=[measure(model,scorer,tok,request,payload['candidate_states'][j],ref,expected,j,clean['choice_char_lengths'][j],permutation,check=count==0 and j==0) for j,request in enumerate(clean['request_pairs'])]
            result=dict(id=chosen['id'],split=chosen['split'],signature=signature,cohort_entry_sha256=digest(chosen),state_file_sha256=state['sha256'],candidates=candidates)
            out.write(json.dumps(result,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno());count+=1;peak=max(peak,torch.cuda.max_memory_allocated())
            save('results/r16_pilot_PROGRESS.json',dict(completed=count,target=64,elapsed_seconds=time.monotonic()-started))
    after=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before==after and count==64
    save('results/r16_pilot_COMPLETE.json',dict(status='complete',n=count,protocol_sha256=sha(LOCK),signature=signature,score_file='results/r16_pilot.jsonl',score_file_sha256=sha('results/r16_pilot.jsonl'),
        head_norm_before=before,head_norm_after=after,weights_modified=False,peak_cuda_allocated_bytes=peak,elapsed_seconds=time.monotonic()-started,model_load_seconds=load_seconds,
        cached_main_states=True,new_main_forwards=0,new_reference_depth=4,KV_cache=False,explicit_attention_mask=False,heldout_accessed=False,standalone_profile=False))
    print('R16_FULL_PILOT_COMPLETE',count,flush=True)


if __name__=='__main__':main()
