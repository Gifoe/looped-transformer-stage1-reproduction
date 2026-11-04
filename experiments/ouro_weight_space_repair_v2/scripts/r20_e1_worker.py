"""One reviewed R20 worker, original uncached call; no label in numerical path."""
import time,os,json
import numpy as np
from r20_e1_common import ROOT,LOCK,read,sha,digest,save,gate

def measure(model,scorer,tok,request,hidden,expected,index,characters,span,permutation,check):
    import torch
    from runtime import encode_pair,check_budget
    from r7_inference import head_logits,forward_states,original_sums,target_lp
    from r7_references import prefix_alignment
    from r7_math import token_logits
    from r20_math import torch_controls,numpy_controls
    from r20_e1_hooks import ValueAblation
    pair=encode_pair(tok,*request);assert hidden['candidate']==index and hidden['target_ids']==pair[1]
    q=head_logits(model,hidden['normalized']);original=original_sums(q,pair[1])
    assert all(v==expected['original_methods'][k]['sum_scores'][index] for k,v in original.items())
    sequence_length=len(pair[0])+len(pair[1])-1
    assert max(span['question_positions']+span['demo_positions'])<len(pair[0])
    exact_receipt=None;call_audit=[]
    from r20_audit import ForwardAudit
    from contextlib import nullcontext
    audit=ForwardAudit(model,scorer,pair[0]+pair[1][:-1]) if check else None
    with (audit if check else nullcontext()):
      with ValueAblation(model,scorer) as hook:
          if check:
              hook.configure(span['question_positions'],False,sequence_length);off=head_logits(model,forward_states(scorer,pair,4));hook.verify(4)
              assert torch.equal(off[1],q[1]) and torch.equal(off[4],q[4])
          hook.configure(span['question_positions'],True,sequence_length);b=head_logits(model,forward_states(scorer,pair,4));bh=hook.verify(4)
          if check:
              hook.configure(span['question_positions'],True,sequence_length);b1=head_logits(model,forward_states(scorer,pair,1));hook.verify(1);assert torch.equal(b1[1],b[1])
          hook.configure(span['demo_positions'],True,sequence_length);d=head_logits(model,forward_states(scorer,pair,4));dh=hook.verify(4)
    if check:
        with ForwardAudit(model,scorer,pair[0]+pair[1][:-1]) as post_audit:
            post=head_logits(model,forward_states(scorer,pair,4))
        assert torch.equal(post[1],q[1]) and torch.equal(post[4],q[4])
        exact_receipt=dict(off_hook_Q1_Q4_fullV_exact=True,post_unload_Q1_Q4_fullV_exact=True,active_B1_independent_depth1_fullV_exact=True,hook_counts_restored=True)
        call_audit=audit.records+post_audit.records
    ref=read('configs/r7_reference_manifest.json')['references']['N0'];rp=encode_pair(tok,ref,request[1]);maps,reason=prefix_alignment(tok,request,pair,[rp]);assert maps is not None,reason
    r=head_logits(model,forward_states(scorer,rp,4));observed=original_sums(r,rp[1]);assert all(v==expected['prior_methods'][0][k]['sum_scores'][index] for k,v in observed.items())
    pieces={k:[] for k in ['primary','demo','raw_demo_matched','shuffle','old_Fisher']};packed={};cpu=None
    for begin in range(0,len(pair[1]),32):
        end=min(begin+32,len(pair[1]));ix=torch.as_tensor(maps[0][begin:end],device=q[4].device)
        args=[x[begin:end] for x in [q[1],q[4],b[1],b[4],d[1],d[4]]];out,diag=torch_controls(*args,permutation)
        old,od=token_logits(args[0],args[1],r[1][ix].unsqueeze(0),r[4][ix].unsqueeze(0));out['old_Fisher']=old;diag['old_Fisher_alpha']=od['alpha']
        if check and begin==0:
            independent,nd=numpy_controls(*(x[:1].cpu().numpy() for x in args),permutation)
            ze=max(float(np.max(np.abs(independent[k]-out[k][:1].cpu().numpy()))) for k in independent);ae=float(np.max(np.abs(nd['alpha']-diag['alpha'][:1].cpu().numpy())))
            assert ze<=2e-5 and ae<=2**-32;cpu=dict(logit_error=ze,alpha_error=ae)
        for k,z in out.items():pieces[k].append(target_lp(z,pair[1][begin:end]))
        for k,v in diag.items():packed.setdefault(k,[]).extend(v.cpu().tolist())
    sums={k:float(torch.cat(v).sum()) for k,v in pieces.items()};late=expected['original_methods']['native_k4']['normalized_scores'][index]-expected['original_methods']['native_k5']['normalized_scores'][index];scores={k:v/int(characters)+late for k,v in sums.items()}
    assert scores['old_Fisher']==expected['M2_scores'][index] and packed['old_Fisher_alpha']==expected['token_diagnostics'][index]['M2_N0_token']['alpha']
    return dict(choice=index,sums=sums,scores=scores,late=late,metrics=packed,CPU_numpy_check=cpu,question_hook=bh,demo_hook=dh,parity_error=0,native_main_parity_error=0,native_N0_parity_error=0,old_M2_score_and_alpha_parity_error=0,first_candidate_exact_checks=exact_receipt,actual_forward_audit=call_audit,reference_positions=maps[0],tokens=len(pair[1]),characters=int(characters),request_sha256=digest(request))

def main():
    started=time.monotonic();spec=gate()
    from r19_common import inputs
    data=inputs(spec);spans={r['id']:r for r in read('configs/r20_SPANS.json')['rows']}
    assert not (ROOT/'results/r20_e1_pilot.jsonl').exists()
    from runtime import load_model,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from weight_edits import tensor_sha256
    import torch
    t=time.monotonic();tok,model=load_model();scorer=OuroCachedScorer(model,5);load_seconds=time.monotonic()-t
    identity=spec['model_identity'];assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for p,h in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+p)==h
    before={n:tensor_sha256(p) for n,p in model.named_parameters()};permutation=np.random.default_rng(2021).permutation(model.config.vocab_size);count=0;peak=0
    torch.cuda.reset_peak_memory_stats()
    with (ROOT/'results/r20_e1_pilot.jsonl').open('x',encoding='utf8') as out:
        for chosen,clean,expected,state in data:
            check_budget(5);assert time.monotonic()<float(os.environ['R20_SOFT_MONOTONIC'])
            payload=torch.load(ROOT/state['file'],map_location='cpu',weights_only=True);assert payload['id']==chosen['id']
            candidates=[measure(model,scorer,tok,req,payload['candidate_states'][j],expected,j,clean['choice_char_lengths'][j],spans[chosen['id']],permutation,count==0 and j==0) for j,req in enumerate(clean['request_pairs'])]
            out.write(json.dumps(dict(id=chosen['id'],split=chosen['split'],candidates=candidates),allow_nan=False)+'\n');out.flush();os.fsync(out.fileno());count+=1;peak=max(peak,torch.cuda.max_memory_allocated());save('results/r20_e1_pilot_PROGRESS.json',dict(completed=count,target=24,elapsed_seconds=time.monotonic()-started))
    after={n:tensor_sha256(p) for n,p in model.named_parameters()};assert before==after and count==24
    save('results/r20_e1_pilot_COMPLETE.json',dict(status='complete',n=count,protocol_sha256=sha(LOCK),score_file_sha256=sha('results/r20_e1_pilot.jsonl'),weights_modified=False,all_parameter_hashes_before=before,all_parameter_hashes_after=after,elapsed_seconds=time.monotonic()-started,model_load_seconds=load_seconds,peak_cuda_allocated_bytes=peak,memory_scope='GB10 unified memory; allocator allocated bytes only',timing_scope='cached-main exploratory pilot, not independent deployment throughput',heldout_accessed=False,KV_cache=False,explicit_attention_mask=False))
if __name__=='__main__':main()
