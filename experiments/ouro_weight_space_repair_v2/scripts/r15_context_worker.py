"""Collect missing vocabulary context contrasts. No repair choice or training."""
import json,os,time
import numpy as np
from r15_common import ROOT,LOCK,read,sha,digest,save,gate,inputs


def measure(model,scorer,tok,request,hidden,refs,expected,index,character_length):
    import torch
    from runtime import encode_pair,check_budget
    from r7_inference import head_logits,forward_states,original_sums,target_lp
    from r7_math import token_logits
    from r7_references import prefix_alignment
    from r15_context_math import context_torch
    pair=encode_pair(tok,*request);assert hidden['candidate']==index and hidden['target_ids']==pair[1]
    q=head_logits(model,hidden['normalized']);original=original_sums(q,pair[1]);maxerror=0.
    for name,value in original.items():
        error=abs(value-expected['original_methods'][name]['sum_scores'][index]);maxerror=max(maxerror,error);assert error==0.,('Main head parity',index,name,error)
    reference_pairs=[encode_pair(tok,context,request[1]) for context in refs]
    maps,reason=prefix_alignment(tok,request,pair,reference_pairs)
    if maps is None:return dict(alignment_supported=False,reason=reason)
    rz=[];reference_sums=[]
    for j,rpair in enumerate(reference_pairs):
        check_budget(5);z=head_logits(model,forward_states(scorer,rpair,4));observed=original_sums(z,rpair[1]);reference_sums.append(observed)
        for name,value in observed.items():
            error=abs(value-expected['prior_methods'][j][name]['sum_scores'][index]);maxerror=max(maxerror,error);assert error==0.,('Reference parity',j,index,name,error)
        rz.append(z)
    packed=dict(change_N1=[],change_N2=[],static_N1=[],static_N2=[],repair_KL=[],correction_energy=[],alpha=[]);parts=[];alphamax=0.
    for start in range(0,len(pair[1]),32):
        stop=min(start+32,len(pair[1]));q1=q[1][start:stop];q4=q[4][start:stop]
        aligned=[{k:v[torch.as_tensor(maps[j][start:stop],device=v.device)] for k,v in z.items()} for j,z in enumerate(rz)]
        znew,diag=token_logits(q1,q4,aligned[0][1].unsqueeze(0),aligned[0][4].unsqueeze(0))
        fixed=q4+.5*(q4-q1);metrics=context_torch(q4,fixed,znew,torch.stack([aligned[1][4],aligned[2][4]]))
        for key,row in [('change_N1',metrics['change'][0]),('change_N2',metrics['change'][1]),('static_N1',metrics['static'][0]),('static_N2',metrics['static'][1]),
                        ('repair_KL',metrics['repair_KL']),('correction_energy',diag['projected_correction_norm2']),('alpha',diag['alpha'])]:packed[key].extend(row.detach().cpu().tolist())
        parts.append(target_lp(znew,pair[1][start:stop]))
    before=expected['token_diagnostics'][index]['M2_N0_token'];assert all(len(v)==len(pair[1]) for v in packed.values())
    alphamax=max(abs(x-y) for x,y in zip(packed['alpha'],before['alpha']));assert alphamax==0.
    total=float(torch.cat(parts).sum());late=expected['original_methods']['native_k4']['normalized_scores'][index]-expected['original_methods']['native_k5']['normalized_scores'][index]
    length=int(character_length);assert length>0
    score=total/length+late;error=abs(score-expected['method_scores']['M2_N0_token'][index]);assert error==0.
    assert max(abs(x-y) for x,y in zip(packed['correction_energy'],before['projected_correction_norm2']))==0.
    return dict(alignment_supported=True,choice=index,characters=length,tokens=len(pair[1]),metrics=packed,original_sums=original,reference_sums=reference_sums,
        M2_score=score,reference_positions=maps,target_ids_sha256=digest(pair[1]),request_sha256=digest(request),
        max_parity_error=max(maxerror,error,alphamax),max_alpha_error=alphamax)


def main():
    started=time.monotonic();spec=gate();data=inputs(spec)
    assert not (ROOT/'results/r15_context_COMPLETE.json').exists() and not (ROOT/'results/r15_context_metrics.jsonl').exists(),'No automatic resume/retry of a probe'
    from runtime import load_model,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from weight_edits import tensor_sha256
    from r15_context_math import synthetic_checks,witness,context_np,context_torch
    import torch
    checks=synthetic_checks();example=witness()
    rng=np.random.default_rng(2021);q=rng.normal(size=(5,13)).astype('float32');f=q+.1;n=f+.05*rng.normal(size=q.shape).astype('float32');r=rng.normal(size=(2,)+q.shape).astype('float32')
    cpu=context_torch(*(torch.from_numpy(x) for x in [q,f,n,r]));reference=context_np(q,f,n,r)
    cpu_error=max(float(np.max(np.abs(cpu[k].numpy()-reference[k]))) for k in cpu);assert cpu_error<1e-12
    save('results/r15_MATH_CHECKS.json',dict(status='passed',checks=checks,nonidentifiability_witness=example,Torch_CPU_NumPy_max_error=cpu_error))
    tok,model=load_model();scorer=OuroCachedScorer(model,5);identity=spec['model_identity']
    assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for name,h in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==h
    before=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight))
    assert before['head']==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    refs=[read('configs/r7_reference_manifest.json')['references'][k] for k in ['N0','N1','N2']]
    signature=digest(dict(protocol_sha256=sha(LOCK),cohort=spec['cohort']));count=0;peak=0;unsupported=0
    with (ROOT/'results/r15_context_metrics.jsonl').open('x',encoding='utf8') as output:
        for chosen,clean,expected,state in data:
            check_budget(15);payload=torch.load(ROOT/state['file'],map_location='cpu',weights_only=True);assert payload['id']==chosen['id']
            torch.cuda.reset_peak_memory_stats();candidates=[]
            for c in [chosen['old_candidate'],chosen['probe_candidate']]:
                candidates.append(measure(model,scorer,tok,clean['request_pairs'][c],payload['candidate_states'][c],refs,expected,c,clean['choice_char_lengths'][c]))
            unsupported+=not all(x['alignment_supported'] for x in candidates)
            result=dict(id=chosen['id'],split=chosen['split'],group=chosen['group'],signature=signature,old_candidate=chosen['old_candidate'],probe_candidate=chosen['probe_candidate'],
                candidates=candidates,state_file_sha256=state['sha256'],cohort_entry_sha256=digest(chosen))
            output.write(json.dumps(result,allow_nan=False)+'\n');output.flush();os.fsync(output.fileno());count+=1
            peak=max(peak,torch.cuda.max_memory_allocated());save('results/r15_context_PROGRESS.json',dict(completed=count,target=58,elapsed_seconds=time.monotonic()-started))
    after=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before==after and count==58
    save('results/r15_context_COMPLETE.json',dict(status='complete',n=58,protocol_sha256=sha(LOCK),signature=signature,score_file='results/r15_context_metrics.jsonl',score_file_sha256=sha('results/r15_context_metrics.jsonl'),
        math_checks_sha256=sha('results/r15_MATH_CHECKS.json'),head_norm_before=before,head_norm_after=after,weights_modified=False,unsupported_questions=unsupported,
        main_hidden_states_reused=True,main_transformer_forwards=0,reference_transformer_depth=4,KV_cache=False,attention_mask_argument=False,
        standalone_deployment_profile=False,peak_cuda_allocated_bytes=peak,elapsed_seconds=time.monotonic()-started,heldout_accessed=False,method_selected=False))
    print('R15_FULL_COHORT_COMPLETE',count,unsupported,flush=True)


if __name__=='__main__':main()
