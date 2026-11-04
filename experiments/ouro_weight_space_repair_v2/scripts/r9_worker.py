"""Actual auxiliary K4 forwards with original BF16/no-mask/no-KV path."""
import argparse,json,os,time
import numpy as np
from r9_core import ROOT,read,sha,digest,jsonlines,save,sources,rows,reverse_prefix,reverse_requests
from r9_scores import score,ELIGIBLE,CONTROL


def summary(records,name):
    base=np.asarray([r['correct']['M0_R6'] for r in records]);new=np.asarray([r['correct'][name] for r in records])
    special=np.asarray([r['correct']['M0_R6'] and not r['correct']['fixed'] for r in records])
    return dict(method=name,n=len(records),correct=int(new.sum()),R6_correct=int(base.sum()),
        rescue=int((new&~base).sum()),damage=int((base&~new).sum()),net=int(new.sum()-base.sum()),
        retained_R6_special=int((new&special).sum()),total_R6_special=int(special.sum()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','development','confirmation','validation','test'],required=True)
    ap.add_argument('--job',required=True);args=ap.parse_args();start=time.time()
    assert args.job.replace('_','').isalnum();protocol=read('configs/r9_PROTOCOL.json');data=rows(args.phase)
    assert sha('data/slcd_fixed_demonstrations.json')==protocol['fixed25_sha256']
    prefix=reverse_prefix(read('data/slcd_fixed_demonstrations.json'))
    assert __import__('hashlib').sha256(prefix.encode()).hexdigest()==protocol['reverse_demonstrations_sha256']
    sig=digest(dict(protocol_sha256=sha('configs/r9_PROTOCOL.json'),source_sha256=sources(),template='reverse_v1',depth=4))
    output=f'results/r9_{args.phase}_{args.job}.jsonl';receipt=output.replace('.jsonl','_COMPLETE.json')
    assert not (ROOT/receipt).exists(),'Completed attempts are immutable'
    done={}
    if (ROOT/output).exists():
        for row in jsonlines(output):assert row['id'] not in done and row['signature']==sig;done[row['id']]=row
    if args.phase=='development':
        pr=read('results/r9_pilot_initial_COMPLETE.json');assert pr['status']=='complete'
        assert sha(pr['score_file'])==pr['score_file_sha256']
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
    tok,model=load_model();scorer=OuroCachedScorer(model,4)
    identity=read('configs/slcd_R6_FINAL_LOCK.json')['model_identity']
    assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for name,value in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==value
    before=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight))
    assert before['head']==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    maxpeak=0;new_questions=0
    for value in data:
        row=value['item']
        if row['id'] in done:continue
        check_budget(30);begin=time.time();torch.cuda.reset_peak_memory_stats();sums={};token_counts=[];target_digests=[]
        clean=dict(question=row['question'],choices=row['choices'])
        requests=reverse_requests(clean,prefix)
        assert reverse_requests({**clean,'gold':-1,'correct':False},prefix)==requests
        per_candidate_token_lp=[]
        for context,continuation in requests:
            pair=encode_pair(tok,context,continuation)
            assert continuation==' '+row['question']
            states=forward_states(scorer,pair,4);logits=head_logits(model,states)
            native=original_sums(logits,pair[1])
            for name,v in native.items():sums.setdefault(name,[]).append(v)
            token_counts.append(len(pair[1]));target_digests.append(digest(pair[1]))
            # Retain full-target scalar LPs for signal localization, never vocab
            # logits/hidden tensors. Fixed is truly logit-level, not LL extrapolation.
            from r7_inference import target_lp
            z1,z4=logits[1],logits[4]
            per_candidate_token_lp.append(dict(native4=target_lp(z4,pair[1]).cpu().tolist(),
                fixed=target_lp(z4+.5*(z4-z1),pair[1]).cpu().tolist()))
        aux4=np.asarray(sums['native_k4'])/len(row['question']);auxF=np.asarray(sums['loopcd_k4_fixed'])/len(row['question'])
        base=value['baseline_scores'];new,diagnostics=score(base['A1'],base['M0_R6'],base['fixed'],aux4,auxF)
        vectors={**base,**{k:v.tolist() for k,v in new.items()},'diagnostic_reverse4':aux4.tolist(),'diagnostic_reverseFixed':auxF.tolist()}
        predictions={k:int(np.argmax(v)) for k,v in vectors.items()};gold=row['gold']
        torch.cuda.synchronize();peak=int(torch.cuda.max_memory_allocated());maxpeak=max(maxpeak,peak)
        rec=dict(id=row['id'],split=value['split'],gold=gold,signature=sig,prompt_sha256=row['prompt_sha256'],
            baseline_scores=base,method_scores={k:v.tolist() for k,v in new.items()},auxiliary_sum_scores=sums,
            auxiliary_normalized_scores=dict(native4=aux4.tolist(),fixed=auxF.tolist()),question_char_denominator=len(row['question']),
            reverse_request_sha256=[digest(p) for p in requests],target_token_counts=token_counts,target_ids_sha256=target_digests,
            auxiliary_token_logprobs=per_candidate_token_lp,predictions=predictions,correct={k:v==gold for k,v in predictions.items()},
            diagnostics=diagnostics,elapsed_seconds=time.time()-begin,peak_cuda_allocated_bytes=peak)
        with (ROOT/output).open('a',encoding='utf8') as f:f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        done[row['id']]=rec;new_questions+=1
        progress=dict(phase=args.phase,job=args.job,completed=len(done),target=len(data),elapsed_seconds=time.time()-start,
            correct={k:sum(r['correct'][k] for r in done.values()) for k in vectors})
        save(f'results/r9_{args.job}_PROGRESS.json',progress)
        if len(done)%16==0 or len(done)==len(data):print(json.dumps(progress),flush=True)
    assert sources()==protocol['source_sha256']
    after=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before==after
    records=[done[v['item']['id']] for v in data]
    table=[summary(records,n) for n in records[0]['predictions']]
    result=dict(status='complete',phase=args.phase,n=len(records),new_questions=new_questions,signature=sig,
        protocol_sha256=sha('configs/r9_PROTOCOL.json'),source_sha256=sources(),score_file=output,score_file_sha256=sha(output),
        summary=table,head_norm_before=before,head_norm_after=after,weights_modified=False,
        elapsed_seconds=time.time()-start,peak_cuda_allocated_bytes=maxpeak,actual_auxiliary_depth=4,
        fixed25_demo_order_preserved=True,primary_R6_prompt_unchanged=True,KV_cache=False,attention_mask_argument=False,
        label_free_scoring=True,control_norm_match_failures=sum(not r['diagnostics']['control_norm_match_possible'] for r in records))
    save(receipt,result);print('R9_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
