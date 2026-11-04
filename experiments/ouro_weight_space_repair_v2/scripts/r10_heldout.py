"""Only frozen shortlisted methods produce held-out ranking/correctness outputs."""
import argparse,json,os,time
import numpy as np
from r10_core import ROOT,read,sha,digest,jsonlines,save,sources,rows,joint_prefix
from r10_auxiliary import infer,selected_scores,baselines
from r9_worker import summary


EXTRA=('r10_auxiliary.py','r10_heldout.py','r10_evaluate.py','r10_chain.py','r10_supervise.py')


def extra_sources():return {n:sha('scripts/'+n) for n in EXTRA}


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['parity','confirmation','validation','test'],required=True);a=p.parse_args();start=time.time()
    phase=a.phase;protocol=read('configs/r10_PROTOCOL.json');data=rows('pilot' if phase=='parity' else phase)
    if phase=='parity':
        names=['joint4_fusion','joint4_protected','jointFixed_protected','balanced4_protected'];data=data[:2]
        lock=read('configs/r10_CANDIDATE_LOCK.json')
    else:
        label={'confirmation':'CANDIDATE','validation':'VALIDATION','test':'FINAL_METHOD'}[phase];lock=read('configs/r10_'+label+'_LOCK.json')
        names=[lock['primary']] if phase=='test' else lock['candidates']
    assert lock['extra_source_sha256']==extra_sources() and lock['source_sha256']==sources()
    assert lock['post_pilot_audit_sha256']==sha('configs/r10_POST_PILOT_AUDIT.json')
    output=f'results/r10_{phase}_selected.jsonl';receipt=output.replace('.jsonl','_COMPLETE.json');assert not (ROOT/receipt).exists()
    sig=digest(dict(protocol_sha256=sha('configs/r10_PROTOCOL.json'),lock_sha256=sha('configs/r10_'+('CANDIDATE' if phase=='parity' else label)+'_LOCK.json'),names=names,extra_source_sha256=extra_sources()))
    done={}
    if (ROOT/output).exists():
        for row in jsonlines(output):assert row['id'] not in done and row['signature']==sig;done[row['id']]=row
    assert set(done)<={v['item']['id'] for v in data}
    from runtime import load_model,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from weight_edits import tensor_sha256
    import torch
    tok,model=load_model();scorer=OuroCachedScorer(model,5);identity=protocol['model_identity']
    assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for n,h in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+n)==h
    before=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before['head']==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    assert sha('data/slcd_fixed_demonstrations.json')==protocol['fixed25_sha256'];prefix=joint_prefix(read('data/slcd_fixed_demonstrations.json'))
    parity={r['id']:r for r in jsonlines('results/r10_pilot_initial.jsonl')} if phase=='parity' else None
    # Validation/test need only the frozen formula's reference set. The other
    # joint-only baselines share its readout; balanced baseline exists only if
    # the shortlist actually requires cyclic forwards.
    balanced=phase=='parity' or 'balanced4_protected' in names;maxerr=0.;maxpeak=0
    for value in data:
        row=value['item']
        if row['id'] in done:continue
        check_budget(30);begin=time.time();torch.cuda.reset_peak_memory_stats()
        clean=dict(question=row['question'],choices=row['choices']);aux=infer(model,scorer,tok,clean,prefix,balanced)
        values,diags=selected_scores(value['baseline_scores'],aux,names);joint=baselines(aux)
        if parity is not None:
            ref=parity[row['id']]
            for n,v in aux['auxiliary_sum_scores'].items():maxerr=max(maxerr,float(np.max(np.abs(np.asarray(v)-np.asarray(ref['auxiliary_sum_scores'][n])))))
            assert aux['balanced_native4_scores']==ref['balanced_native4_scores']
            for n in names:maxerr=max(maxerr,float(np.max(np.abs(values[n]-np.asarray(ref['method_scores'][n])))))
            assert maxerr==0.,('Heldout path parity',row['id'],maxerr)
        vectors={**value['baseline_scores'],**{k:v.tolist() for k,v in values.items()},**{k:v.tolist() for k,v in joint.items()}}
        preds={k:int(np.argmax(v)) for k,v in vectors.items()};gold=row['gold'];torch.cuda.synchronize();peak=int(torch.cuda.max_memory_allocated());maxpeak=max(maxpeak,peak)
        rec=dict(id=row['id'],split=value['split'],gold=gold,signature=sig,prompt_sha256=row['prompt_sha256'],baseline_scores=value['baseline_scores'],
            method_scores={k:v.tolist() for k,v in values.items()},joint_baseline_scores={k:v.tolist() for k,v in joint.items()},
            predictions=preds,correct={k:v==gold for k,v in preds.items()},diagnostics=diags,**aux,elapsed_seconds=time.time()-begin,peak_cuda_allocated_bytes=peak)
        with (ROOT/output).open('a',encoding='utf8') as f:f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        done[row['id']]=rec;progress=dict(phase=phase,completed=len(done),target=len(data),elapsed_seconds=time.time()-start,correct={k:sum(r['correct'][k] for r in done.values()) for k in vectors})
        save('results/r10_'+phase+'_PROGRESS.json',progress)
        if len(done)%50==0 or len(done)==len(data):print(json.dumps(progress),flush=True)
    after=dict(head=tensor_sha256(model.lm_head.weight),norm=tensor_sha256(model.model.norm.weight));assert before==after
    assert lock['extra_source_sha256']==extra_sources() and protocol['source_sha256']==sources()
    records=[done[v['item']['id']] for v in data]
    result=dict(status='complete',phase=phase,n=len(records),names=names,signature=sig,score_file=output,score_file_sha256=sha(output),
        protocol_sha256=sha('configs/r10_PROTOCOL.json'),source_sha256=sources(),extra_source_sha256=extra_sources(),
        summary=[summary(records,k) for k in records[0]['predictions']],head_norm_before=before,head_norm_after=after,weights_modified=False,
        max_parity_error=maxerr if phase=='parity' else None,balanced_forwards=balanced,KV_cache=False,attention_mask_argument=False,
        elapsed_seconds=time.time()-start,peak_cuda_allocated_bytes=maxpeak,original_R6_scores_reused=True,actual_auxiliary_depth=5)
    save(receipt,result);print('R10_SELECTED_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
