"""Joint-option K5 auxiliary trajectory, original uncached model attention call."""
import argparse,json,os,time
import numpy as np
from r10_core import ROOT,read,sha,digest,jsonlines,save,sources,rows,joint_prefix,joint_requests
from r10_scores import score,ELIGIBLE
from r9_worker import summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','development','confirmation','validation','test'],required=True)
    ap.add_argument('--job',required=True);args=ap.parse_args();started=time.time();assert args.job.replace('_','').isalnum()
    protocol=read('configs/r10_PROTOCOL.json');data=rows(args.phase)
    assert sha('data/slcd_fixed_demonstrations.json')==protocol['fixed25_sha256']
    prefix=joint_prefix(read('data/slcd_fixed_demonstrations.json'))
    assert __import__('hashlib').sha256(prefix.encode()).hexdigest()==protocol['joint_demonstrations_sha256']
    sig=digest(dict(protocol_sha256=sha('configs/r10_PROTOCOL.json'),source_sha256=sources(),template='joint_v1',depth=5))
    output=f'results/r10_{args.phase}_{args.job}.jsonl';receipt=output.replace('.jsonl','_COMPLETE.json')
    assert not (ROOT/receipt).exists(),'Completed attempts are immutable';done={}
    if (ROOT/output).exists():
        for row in jsonlines(output):assert row['id'] not in done and row['signature']==sig;done[row['id']]=row
    if args.phase=='development':
        pr=read('results/r10_pilot_initial_COMPLETE.json');assert pr['status']=='complete' and sha(pr['score_file'])==pr['score_file_sha256']
        for row in jsonlines(pr['score_file']):
            assert row['signature']==sig
            if row['id'] not in done:
                with (ROOT/output).open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                done[row['id']]=row
    assert set(done)<={v['item']['id'] for v in data}
    from runtime import load_model,encode_pair,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from r7_inference import forward_states,head_logits,original_sums
    from weight_edits import tensor_sha256
    import torch
    tok,model=load_model();scorer=OuroCachedScorer(model,5)
    identity=read('configs/slcd_R6_FINAL_LOCK.json')['model_identity']
    assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for name,value in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==value
    before=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight))
    assert before['head']==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    maxpeak=0;new_questions=0;forwards=0;depth_check=None
    for value in data:
        row=value['item']
        if row['id'] in done:continue
        check_budget(30);begin=time.time();torch.cuda.reset_peak_memory_stats();sums={}
        clean=dict(question=row['question'],choices=row['choices']);requests=joint_requests(clean,prefix)
        assert joint_requests(dict(clean,gold=-1,correct=False),prefix)==requests
        pairs=[encode_pair(tok,*p) for p in requests];memo={};target_counts=[];input_counts=[]
        def score_pairs(encoded):
            scored={}
            for pair in encoded:
                # Exact duplicate MODEL INPUT+positions only; no cross-context KV
                # cache. Single-token letter answers usually share one full input.
                pre,tar=pair;positions=tuple(range(len(pre)-1,len(pre)+len(tar)-1));key=(tuple(pre+tar[:-1]),positions)
                if key not in memo:memo[key]=head_logits(model,forward_states(scorer,pair,5))
                native=original_sums(memo[key],tar)
                for name,v in native.items():scored.setdefault(name,[]).append(v)
                target_counts.append(len(tar));input_counts.append(len(key[0]))
            return scored
        sums=score_pairs(pairs);rotations=[dict(order=list(range(len(pairs))),sum_scores=sums)]
        if not done:
            independent=head_logits(model,forward_states(scorer,pairs[0],4))
            independent_sums=original_sums(independent,pairs[0][1]);forwards+=1
            err=max(abs(independent_sums[k]-sums[k][0]) for k in independent_sums)
            assert err==0.,('Independent auxiliary K4 vs shared K5 trajectory',err)
            depth_check=dict(status='passed',id=row['id'],candidate=0,max_error=err)
            scorer.kmax=5;scorer._set_depth(5)
        for shift in range(1,len(pairs)):
            order=[(i+shift)%len(pairs) for i in range(len(pairs))]
            rotated=dict(question=clean['question'],choices=[clean['choices'][i] for i in order])
            rq=joint_requests(rotated,prefix);rp=[encode_pair(tok,*p) for p in rq];rs=score_pairs(rp)
            mapped={}
            for name,vector in rs.items():
                values=np.empty(len(pairs))
                for j,i in enumerate(order):values[i]=vector[j]
                mapped[name]=values.tolist()
            rotations.append(dict(order=order,sum_scores=mapped,request_sha256=[digest(p) for p in rq]))
        balanced4=np.asarray([r['sum_scores']['native_k4'] for r in rotations]).mean(axis=0)
        forwards+=len(memo)
        base=value['baseline_scores'];new,diagnostics=score(base['A1'],base['M0_R6'],base['fixed'],sums['native_k4'],sums['loopcd_k4_fixed'],balanced4)
        auxdiag={'diagnostic_joint4':sums['native_k4'],'diagnostic_jointFixed':sums['loopcd_k4_fixed'],
                 'diagnostic_jointA1':(np.asarray(sums['loopcd_k4_fixed'])+np.asarray(sums['native_k4'])-np.asarray(sums['native_k5'])).tolist(),
                 'diagnostic_balanced4':balanced4.tolist()}
        vectors={**base,**{k:v.tolist() for k,v in new.items()},**auxdiag};predictions={k:int(np.argmax(v)) for k,v in vectors.items()};gold=row['gold']
        torch.cuda.synchronize();peak=int(torch.cuda.max_memory_allocated());maxpeak=max(maxpeak,peak)
        rec=dict(id=row['id'],split=value['split'],gold=gold,signature=sig,prompt_sha256=row['prompt_sha256'],baseline_scores=base,
            method_scores={k:v.tolist() for k,v in new.items()},auxiliary_sum_scores=sums,cyclic_rotations=rotations,
            balanced_native4_scores=balanced4.tolist(),common_auxiliary_char_denominator=1,
            auxiliary_request_sha256=[digest(p) for p in requests],target_token_counts=target_counts,target_ids_sha256=[digest(p[1]) for p in pairs],
            model_input_token_counts=input_counts,unique_auxiliary_forwards=len(memo),predictions=predictions,correct={k:v==gold for k,v in predictions.items()},
            diagnostics=diagnostics,elapsed_seconds=time.time()-begin,peak_cuda_allocated_bytes=peak)
        with (ROOT/output).open('a',encoding='utf8') as f:f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        done[row['id']]=rec;new_questions+=1;progress=dict(phase=args.phase,job=args.job,completed=len(done),target=len(data),elapsed_seconds=time.time()-started,
            correct={k:sum(r['correct'][k] for r in done.values()) for k in vectors})
        save(f'results/r10_{args.job}_PROGRESS.json',progress)
        if len(done)%16==0 or len(done)==len(data):print(json.dumps(progress),flush=True)
    assert sources()==protocol['source_sha256'];after=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before==after
    records=[done[v['item']['id']] for v in data]
    result=dict(status='complete',phase=args.phase,n=len(records),new_questions=new_questions,unique_auxiliary_forwards_new=forwards,signature=sig,
        protocol_sha256=sha('configs/r10_PROTOCOL.json'),source_sha256=sources(),score_file=output,score_file_sha256=sha(output),
        summary=[summary(records,k) for k in records[0]['predictions']],head_norm_before=before,head_norm_after=after,weights_modified=False,
        elapsed_seconds=time.time()-started,peak_cuda_allocated_bytes=maxpeak,actual_auxiliary_depth=5,
        primary_R6_prompt_unchanged=True,fixed25_demo_ids_order_preserved=True,KV_cache=False,attention_mask_argument=False,label_free_scoring=True,
        independent_depth_check=depth_check,
        control_norm_match_failures={name:sum(not r['diagnostics'][name]['control_norm_match_possible'] for r in records) for name in ['joint4','jointFixed','balanced4']})
    save(receipt,result);print('R10_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
