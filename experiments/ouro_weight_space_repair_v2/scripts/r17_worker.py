"""Actual paired cue interventions, no new decoder or gold-based selection."""
import json,os,time
import numpy as np
from r17_common import ROOT,LOCK,read,sha,digest,save,gate,inputs


def main():
    started=time.monotonic();spec=gate();data=inputs(spec)
    assert not any((ROOT/p).exists() for p in ['results/r17_paired_responses.jsonl','results/r17_paired_COMPLETE.json']),'No automatic retry/resume'
    from r17_math import synthetic_checks,torch_stats,numpy_stats
    save('results/r17_MATH_CHECKS.json',synthetic_checks())
    from runtime import load_model,encode_pair,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from r7_inference import head_logits,forward_states,original_sums
    from r7_references import prefix_alignment
    from weight_edits import tensor_sha256
    import torch
    tok,model=load_model();scorer=OuroCachedScorer(model,4);identity=spec['model_identity']
    assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for p,h in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+p)==h
    before=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before['head']==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    permutation=np.random.default_rng(2021).permutation(model.config.vocab_size);n0=read('configs/r7_reference_manifest.json')['references']['N0'];assert n0=='Answer:'
    signature=digest(dict(protocol_sha256=sha(LOCK),cohort=spec['cohort']));count=0;peak=0
    with (ROOT/'results/r17_paired_responses.jsonl').open('x',encoding='utf8') as out:
        for chosen,clean,expected,st in data:
            check_budget(5);t=time.monotonic();payload=torch.load(ROOT/st['file'],map_location='cpu',weights_only=True);hidden=payload['candidate_states'][chosen['candidate']]
            pair=encode_pair(tok,*clean['request']);assert payload['id']==chosen['id'] and hidden['candidate']==chosen['candidate'] and hidden['target_ids']==pair[1]
            q=head_logits(model,hidden['normalized']);observed=original_sums(q,pair[1]);assert observed==expected['main_sums']
            requests=[clean['changed_request'],[n0,clean['request'][1]],['Response:',clean['request'][1]]]
            pairs=[encode_pair(tok,*request) for request in requests];maps,reason=prefix_alignment(tok,clean['request'],pair,pairs);assert maps is not None,('No filtered-prefix analysis',reason)
            torch.cuda.reset_peak_memory_stats();fresh=[];timings=[]
            for rp in pairs:
                check_budget(3);ts=time.monotonic();fresh.append(head_logits(model,forward_states(scorer,rp,4)));torch.cuda.synchronize();timings.append(time.monotonic()-ts)
            assert original_sums(fresh[1],pairs[1][1])==expected['reference_sums']
            packed={};cpu_check=None
            for begin in range(0,len(pair[1]),32):
                end=min(begin+32,len(pair[1]));aligned=[{k:z[k][torch.as_tensor(m[begin:end],device=z[k].device)] for k in [1,4]} for z,m in zip(fresh,maps)]
                sq,n,sn=aligned;values=[q[1][begin:end],q[4][begin:end],n[1],n[4],sq[1],sq[4],sn[1],sn[4]]
                stats=torch_stats(*values,permutation=permutation)
                if count==0 and begin==0:
                    reference=numpy_stats(*(v[:1].cpu().numpy() for v in values),permutation=permutation);error=0.
                    for k,v in reference.items():
                        actual=stats[k][:1].cpu().numpy()
                        if v.dtype==bool:assert np.array_equal(v,actual)
                        else:error=max(error,float(np.max(np.abs(v-actual))))
                    assert error<1e-9;cpu_check=dict(real_NumPy_GPU_max_error=error)
                for k,v in stats.items():packed.setdefault(k,[]).extend(v.cpu().tolist())
            assert all(len(v)==len(pair[1]) for v in packed.values())
            record=dict(id=chosen['id'],split=chosen['split'],candidate=chosen['candidate'],signature=signature,cohort_entry_sha256=digest(chosen),state_file_sha256=st['sha256'],
                request_sha256=digest(clean['request']),changed_requests_sha256=[digest(r) for r in requests],target_ids_sha256=digest(pair[1]),reference_positions=maps,
                tokens=len(pair[1]),characters=clean['characters'],metrics=packed,CPU_check=cpu_check,main_reference_parity_error=0.,
                seconds=time.monotonic()-t,fresh_forward_seconds=dict(changed_real=timings[0],original_N0=timings[1],changed_N0=timings[2]))
            out.write(json.dumps(record,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno());count+=1;peak=max(peak,torch.cuda.max_memory_allocated())
            save('results/r17_paired_PROGRESS.json',dict(completed=count,target=12,elapsed_seconds=time.monotonic()-started))
    after=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before==after and count==12
    save('results/r17_paired_COMPLETE.json',dict(status='complete',n=count,protocol_sha256=sha(LOCK),signature=signature,score_file='results/r17_paired_responses.jsonl',score_file_sha256=sha('results/r17_paired_responses.jsonl'),
        head_norm_before=before,head_norm_after=after,weights_modified=False,peak_cuda_allocated_bytes=peak,elapsed_seconds=time.monotonic()-started,heldout_accessed=False,
        cached_original_main=True,new_main_depth=4,new_reference_depth=4,KV_cache=False,explicit_attention_mask=False,new_decoding_accuracy=False,standalone_profile=False))
    print('R17_COMPLETE_OBSERVATION',count,flush=True)


if __name__=='__main__':main()
