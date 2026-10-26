"""Three exact paper Logits configurations sharing one matched Ouro trajectory."""
import hashlib
import importlib.metadata
import json
import time
import torch
from common import ROOT,BASE,CFG,PROTOCOL_SHA,Budget,load,read,save,append,existing,encode_pair,candidate_result
from methods import adaptive_logits


def score(tok,model,row):
    sums={key:[] for key in CFG['arc_methods']};counts=[];strengths=[]
    torch.cuda.synchronize();started=time.monotonic()
    for prompt,answer in row['request_pairs']:
        prefix,target=encode_pair(tok,prompt,answer)
        x=torch.tensor([prefix+target[:-1]],device='cuda');target=torch.tensor(target,device='cuda')
        _,states,_=model.model(input_ids=x,use_cache=False)
        assert len(states)==4
        selection=slice(len(prefix)-1,None)
        early=model.lm_head(states[0][:,selection,:])[0].float()
        final=model.lm_head(states[3][:,selection,:])[0].float()
        fixed=final+CFG['arc_fixed_omega']*(final-early)
        adaptive,strength=adaptive_logits(final,early,CFG['arc_adaptive_cap'])
        assert bool(torch.isfinite(adaptive).all())
        top=final.softmax(-1).topk(2,dim=-1).values
        independent_strength=1-(top[:,0]-top[:,1])
        assert torch.equal(strength,independent_strength)
        assert torch.allclose(adaptive,(1+independent_strength[:,None])*final-independent_strength[:,None]*early,atol=2e-4,rtol=1e-5)
        for method,z in [('baseline',final),('fixed',fixed),('adaptive',adaptive)]:
            sums[method].append(float(z.log_softmax(-1).gather(-1,target[:,None]).sum()))
        counts.append(len(target));strengths.extend(strength.cpu().tolist())
    torch.cuda.synchronize()
    return dict(id=row['id'],task=row['task'],source_index=row['source_index'],gold=row['gold'],
        prompt_sha256=row['prompt_sha256'],fewshot_count=25,primary_metric='acc_norm',model='Ouro-1.4B',
        model_revision=CFG['model_revision'],methods={key:candidate_result(values,row['choice_char_lengths'],counts,row['gold'],'acc_norm') for key,values in sums.items()},
        choice_token_counts=counts,choice_char_lengths=row['choice_char_lengths'],
        fixed_omega=.5,adaptive_cap=1.,adaptive_strength_min=min(strengths),adaptive_strength_max=max(strengths),
        recurrent_depth=4,reference_depth=1,elapsed_s=time.monotonic()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),
        compute_scope='One four-loop teacher-forced trajectory per candidate, shared exactly across all three readouts; not three independent generation timings')


def parity(tok,model):
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.instance import Instance
    model.early_exit_step=3
    reference=HFLM(pretrained=model,tokenizer=tok,backend='causal',batch_size=1,device='cuda',max_length=CFG['max_context_tokens'],add_bos_token=False,dtype='bfloat16',trust_remote_code=True)
    cases=[]
    for row in read('data/arc_parity_validation.jsonl'):
        result=score(tok,model,row)
        requests=[Instance(request_type='loglikelihood',doc=row['source_doc'],arguments=tuple(pair),idx=i,metadata=(row['task'],row['source_index'],1)) for i,pair in enumerate(row['request_pairs'])]
        expected=reference.loglikelihood(requests,disable_tqdm=True)
        actual=result['methods']['baseline']
        errors=[abs(ll[0]-value)/count for ll,value,count in zip(expected,actual['sum_scores'],result['choice_token_counts'])]
        assert max(errors)<=.05
        import numpy as np
        assert int((np.asarray([e[0] for e in expected])/row['choice_char_lengths']).argmax())==actual['normalized_prediction']
        cases.append(dict(id=row['id'],max_error_per_token=max(errors),baseline_prediction_matches=True))
    # Native final readout and zero strength must preserve this checkpoint.
    x=tok('The Earth orbits the Sun.',return_tensors='pt').to('cuda')
    _,states,_=model.model(**x,use_cache=False)
    expected=model(**x,use_cache=False,exit_at_step=3,logits_to_keep=1).logits
    final=model.lm_head(states[-1][:,-1:,:]);early=model.lm_head(states[0][:,-1:,:])
    assert torch.equal(expected,final)
    assert torch.equal(adaptive_logits(final,early,0)[0],final.float())
    report=dict(passed=True,cases=cases,native_final_readout_exact=True,zero_guidance_exact=True,
        checkpoint_revision=CFG['model_revision'],protocol_sha256=PROTOCOL_SHA,harness_version=importlib.metadata.version('lm_eval'))
    save('results/ARC_PARITY.json',report)


def main():
    assert importlib.metadata.version('lm_eval')==CFG['lm_eval_version']=='0.4.9'
    audit=json.loads((ROOT/'data/FULL_MANIFEST_AUDIT.json').read_text())
    path=ROOT/'data/arc_full_test.jsonl'
    assert audit['passed'] and audit['protocol_sha256']==PROTOCOL_SHA
    assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['full_test']['sha256']
    rows=read(path);assert len(rows)==1172
    assert [r['source_index'] for r in rows]==list(range(1172))
    with Budget('full_ARC_LoopCD_Ouro1.4B') as budget:
        tok,model=load('Ouro-1.4B')
        info=json.loads((ROOT/'results/model_load_Ouro-1.4B.json').read_text())
        assert info['revision']==CFG['model_revision']
        assert all(not info['loading_info'].get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        parity(tok,model)
        output='trajectories/arc_three_methods.jsonl';done=existing(output)
        torch.cuda.reset_peak_memory_stats()
        for row in rows:
            if row['id'] in done:continue
            budget.check(reserve=60)
            if __import__('shutil').disk_usage(ROOT).free<5*1024**3:raise RuntimeError('5GiB reserve reached')
            result=score(tok,model,row);append(output,result);done[row['id']]=result
            save('STATUS.json',dict(stage='arc_full',completed=len(done),expected=1172))
            if len(done)%10==0:print(json.dumps(dict(completed=len(done),expected=1172)),flush=True)
        assert set(done)=={r['id'] for r in rows}
        save('results/ARC_WORKER_COMPLETE.json',dict(passed=True,questions=1172,readouts=3516,protocol_sha256=PROTOCOL_SHA))


if __name__=='__main__':main()
