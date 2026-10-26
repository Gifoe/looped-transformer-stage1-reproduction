"""Actual HF/harness and official-source parity, before new formal predictions."""
import argparse
from collections import Counter
import hashlib
import json
import time
import numpy as np
import torch
from common import ROOT,BASE,CFG,Budget,load,read,save,encode_pair,raw_states
from methods import dola_scores,qwen_premature,official_dola_oracle,adaptive_logits,projection_controls,native_cdf
from qwen_recipe import Recipe

def difference(a,b):
    a=a.float();b=b.float();delta=(a-b).abs()
    return dict(max_abs=float(delta.max()),relative_l2=float(delta.norm()/a.norm().clamp_min(1e-8)),argmax_agreement=float((a.argmax(-1)==b.argmax(-1)).float().mean()))

def harness_check(tok,model,rows,name):
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.instance import Instance
    if name.startswith('Ouro'):model.early_exit_step=3
    reference=HFLM(pretrained=model,tokenizer=tok,backend='causal',batch_size=1,device='cuda',max_length=CFG['max_context_tokens'],add_bos_token=False,dtype='bfloat16',trust_remote_code=True)
    cases=[]
    for row in rows:
        requests=[Instance(request_type='loglikelihood',doc=row['source_doc'],arguments=tuple(pair),idx=i,metadata=(row['task'],row['source_index'],1)) for i,pair in enumerate(row['request_pairs'])]
        expected=reference.loglikelihood(requests,disable_tqdm=True);actual=[];errors=[]
        for pair,(expected_ll,_) in zip(row['request_pairs'],expected):
            prefix,target=encode_pair(tok,*pair)
            x=torch.tensor([prefix+target[:-1]],device='cuda');selection=torch.arange(len(prefix)-1,x.shape[1],device='cuda')
            z=model(input_ids=x,use_cache=False,logits_to_keep=selection).logits[0].float()
            ll=float(z.log_softmax(-1).gather(-1,torch.tensor(target,device='cuda')[:,None]).sum())
            actual.append(ll);errors.append(abs(ll-expected_ll)/len(target))
        assert max(errors)<=.05,(name,row['id'],errors)
        normalized=np.asarray(actual)/row['choice_char_lengths'];ref_normalized=np.asarray([v[0] for v in expected])/row['choice_char_lengths']
        assert int(np.argmax(actual))==int(np.argmax([v[0] for v in expected]))
        assert int(normalized.argmax())==int(ref_normalized.argmax())
        cases.append(dict(id=row['id'],max_error_per_token=max(errors),harness_sum_loglikelihood=[v[0] for v in expected],selected_sum_loglikelihood=actual,prompt_tokens=len(tok.encode(row['prompt'],add_special_tokens=False)),n_choices=len(actual)))
    return dict(passed=True,harness_version='0.4.9',cases=cases,force_ouro_final_depth=name.startswith('Ouro'))

def qwen_checks(tok,model):
    recipe=Recipe(model);identities=[]
    for text in ['Question: What is 2 + 2?\nAnswer:','The Earth orbits the Sun.']:
        x=tok(text,return_tensors='pt').to('cuda')
        for cache in [False,True]:
            recipe.restore();base=model(**x,use_cache=cache,logits_to_keep=1)
            for config in [dict(strategy='naive',K=1),dict(strategy='euler',K=1),dict(strategy='block_anchored',K=1,anchor_beta=.5),dict(strategy='block_anchored',K=6,anchor_beta=1)]:
                recipe.set(config);patched=model(**x,use_cache=cache,logits_to_keep=1)
                diff=difference(base.logits,patched.logits)
                assert diff['max_abs']<=.125 and diff['relative_l2']<=.01 and diff['argmax_agreement']==1
                if cache:assert all(patched.past_key_values.get_seq_length(i)==x.input_ids.shape[1] for i in range(36))
                identities.append(dict(config=config,use_cache=cache,error=diff))
    recipe.restore();dola=[]
    layers=CFG['dola']['low_layers']+CFG['dola']['high_layers']
    for prompt,answer in [('Question: Which planet is closest to the Sun?\nAnswer:',' Mercury'),('Question: What do green plants need?\nAnswer:',' sunlight and water')]:
        prefix,targets=encode_pair(tok,prompt,answer);x=torch.tensor([prefix+targets[:-1]],device='cuda');selection=slice(len(prefix)-1,None)
        with qwen_premature(model,selection,layers) as captured:out=model.model(input_ids=x,use_cache=False,output_hidden_states=True)
        for layer in layers:assert torch.equal(captured[layer],out.hidden_states[layer][:,selection,:]),layer
        final=model.lm_head(out.last_hidden_state[:,selection,:])[0].float()
        for label,group in [('low',CFG['dola']['low_layers']),('high',CFG['dola']['high_layers'])]:
            early=torch.stack([model.lm_head(captured[layer])[0].float() for layer in group])
            scores,selected=dola_scores(final,early,group)
            target=torch.tensor(targets,device='cuda');actual=float(scores.gather(-1,target[:,None]).sum())
            expected,histogram=official_dola_oracle(final,early,group,target)
            assert abs(expected-actual)<=1e-4,(label,expected,actual)
            assert Counter(selected)==Counter({key:value for key,value in histogram.items() if value})
            dola.append(dict(group=label,absolute_score_error=abs(expected-actual),selected_layers=selected,reference_layers=histogram))
    return dict(passed=True,identities=identities,dola_source_kernel=dola,source_sha256=hashlib.sha256((ROOT/'sources/dola.py').read_bytes()).hexdigest(),intermediate_states='raw, matches source and HF hidden_states indices')

def ouro_checks(tok,model,name):
    cases=[];gate_checks=[]
    for text in ['Question: What is 2 + 2?\nAnswer:','Water freezes at zero degrees Celsius.']:
        x=tok(text,return_tensors='pt').to('cuda')
        with raw_states(model,slice(-1,None)) as raw:_,states,gates=model.model(**x,use_cache=False)
        assert len(states)==len(raw)==4
        for r in range(4):
            native=model(**x,use_cache=False,exit_at_step=r,logits_to_keep=1).logits
            actual=model.lm_head(states[r][:,-1:,:]);assert torch.equal(native,actual)
            assert torch.equal(actual,model.lm_head(model.model.norm(raw[r])))
        for threshold in [.01,.25,.45,.5]:
            cdf=native_cdf(gates);index=next((i for i,v in enumerate(cdf) if bool(v>=threshold)),3)
            model.early_exit_step=None
            expected=model(**x,use_cache=False,exit_threshold=threshold,logits_to_keep=1).logits
            assert torch.equal(expected,model.lm_head(states[index][:,-1:,:]))
            gate_checks.append(dict(threshold=threshold,depth=index+1,max_abs=0))
        cached=model(**x,use_cache=True,exit_at_step=3,logits_to_keep=1)
        layer_count=model.config.num_hidden_layers*4
        assert all(cached.past_key_values.get_seq_length(i)==x.input_ids.shape[1] for i in range(layer_count))
        token=cached.logits[:,-1,:].argmax(-1,keepdim=True)
        incremental=model(input_ids=token,past_key_values=cached.past_key_values,use_cache=True,exit_at_step=3,logits_to_keep=1).logits
        whole=model(input_ids=torch.cat([x.input_ids,token],1),use_cache=False,exit_at_step=3,logits_to_keep=1).logits
        diff=difference(whole,incremental);assert diff['relative_l2']<.02 and diff['argmax_agreement']==1
        cases.append(dict(cache_layers=layer_count,cache_error=diff))
    # Source formula properties at actual model logits, independent of labels.
    final=model.lm_head(states[-1][:,-1:,:])[0].float();early=model.lm_head(states[0][:,-1:,:])[0].float()
    zero,_=adaptive_logits(final,early,0);assert torch.equal(zero,final)
    guided,strength=adaptive_logits(final,early,1);assert bool(((strength>=0)&(strength<=1)).all())
    parallel,orthogonal,pu,ou=projection_controls(final,early,.5)
    centered=final-final.mean(-1,keepdim=True)
    expected=(final+.5*(final-early)).log_softmax(-1)
    actual=(parallel+orthogonal-centered).log_softmax(-1)
    projection_error=float((expected-actual).abs().max());assert projection_error<1e-4
    relative_inner=float((pu*ou).sum().abs()/(pu.norm()*ou.norm()).clamp_min(1e-20));assert relative_inner<1e-4
    model.early_exit_step=3
    return dict(passed=True,cases=cases,native_gate=gate_checks,projection_log_probability_max_error=projection_error,projection_relative_inner_product=relative_inner,layers=model.config.num_hidden_layers,loops=4,cache_layers=model.config.num_hidden_layers*4,source_sha256=hashlib.sha256((BASE/'models'/name/'modeling_ouro.py').read_bytes()).hexdigest())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=CFG['models']);args=parser.parse_args()
    with Budget('parity_'+args.model):
        tok,model=load(args.model)
        report=qwen_checks(tok,model) if args.model.startswith('Qwen') else ouro_checks(tok,model,args.model)
        rows=read('data/validation.jsonl');subset=[next(r for r in rows if r['task']==task or task=='mmlu' and r['task'].startswith('mmlu_')) for task in ['arc_challenge','openbookqa','mmlu']]
        report['harness']=harness_check(tok,model,subset,args.model)
        report['peak_memory_bytes']=torch.cuda.max_memory_allocated()
        save('results/parity_'+args.model+'.json',report)
        print(json.dumps(dict(model=args.model,passed=True,peak_memory_bytes=report['peak_memory_bytes'])),flush=True)

if __name__=='__main__':main()
