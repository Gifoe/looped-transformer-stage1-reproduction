"""Paired likelihood requests; shared Ouro trajectory and separate Qwen methods."""
import argparse
from collections import Counter
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,BASE,CFG,Budget,load,read,save,append,existing,encode_pair,vocab_stats,candidate_result,raw_states,hidden_change
from methods import adaptive_logits,projection_controls,dola_scores,qwen_premature,native_cdf
from qwen_recipe import Recipe

def signature(row):
    return dict(id=row['id'],task=row['task'],gold=row['gold'],prompt_sha256=row['prompt_sha256'],fewshot_count=row['fewshot_count'],primary_metric=row['primary_metric'])

def qwen_score(tok,model,recipe,row,method):
    torch.cuda.synchronize();start=time.monotonic();sums=[];token_counts=[];selected_layers=Counter();prompt_stats=None
    layers=CFG['dola']['low_layers' if method=='dola_low' else 'high_layers'] if method.startswith('dola_') else []
    for option,(prompt,continuation) in enumerate(row['request_pairs']):
        prefix,answer=encode_pair(tok,prompt,continuation)
        x=torch.tensor([prefix+answer[:-1]],device='cuda');target=torch.tensor(answer,device='cuda');selection=slice(len(prefix)-1,None)
        if layers:
            with qwen_premature(model,selection,layers) as states:
                output=model.model(input_ids=x,use_cache=False)
            z=model.lm_head(output.last_hidden_state[:,selection,:])[0].float()
            early=torch.stack([model.lm_head(states[layer])[0].float() for layer in layers])
            scores,picked=dola_scores(z,early,layers);selected_layers.update(picked)
        else:
            recipe.position=len(prefix)-1
            select=torch.arange(len(prefix)-1,x.shape[1],device='cuda')
            z=model(input_ids=x,use_cache=False,logits_to_keep=select).logits[0].float();scores=z.log_softmax(-1)
        sums.append(float(scores.gather(-1,target[:,None]).sum()));token_counts.append(len(answer))
        if option==0:prompt_stats,_=vocab_stats(z[0])
    torch.cuda.synchronize()
    return dict(**signature(row),model='Qwen3-4B-Base',method=method,
                result=candidate_result(sums,row['choice_char_lengths'],token_counts,row['gold'],row['primary_metric'],proper_probability=not layers),
                full_vocab_stats=prompt_stats,selected_premature_layers=dict(selected_layers),
                elapsed_s=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated())

def ouro_score(tok,model,row,name):
    methods=[f'r{depth}' for depth in range(1,5)]+['adaptive_paper','parallel_05','orthogonal_05']
    methods += [f'{space}_{strength:g}' for strength in CFG['loopcd_strength_grid'] for space in ['logits','hidden']]
    methods += [f'temperature_{value:g}' for value in CFG['temperature_grid']]
    sums={method:[] for method in methods};token_counts=[];trajectory=[];geometry=None;adaptive_strengths=[]
    torch.cuda.synchronize();start=time.monotonic()
    for option,(prompt,continuation) in enumerate(row['request_pairs']):
        prefix,answer=encode_pair(tok,prompt,continuation)
        x=torch.tensor([prefix+answer[:-1]],device='cuda');target=torch.tensor(answer,device='cuda');selection=slice(len(prefix)-1,None)
        with raw_states(model,selection) as raw:_,states,gates=model.model(input_ids=x,use_cache=False)
        selected=[state[:,selection,:] for state in states]
        logits=[model.lm_head(state)[0].float() for state in selected]
        def collect(key,z):sums[key].append(float(z.float().log_softmax(-1).gather(-1,target[:,None]).sum()))
        for depth,z in enumerate(logits,1):collect('r'+str(depth),z)
        for strength in CFG['loopcd_strength_grid']:
            collect(f'logits_{strength:g}',logits[-1]+strength*(logits[-1]-logits[0]))
            hidden=raw[-1]+strength*(raw[-1]-raw[0])
            collect(f'hidden_{strength:g}',model.lm_head(model.model.norm(hidden))[0])
        guided,omega=adaptive_logits(logits[-1],logits[0],CFG['loopcd_mc_adaptive_cap']);collect('adaptive_paper',guided)
        adaptive_strengths.extend(omega.cpu().tolist())
        parallel,orthogonal,parallel_update,orthogonal_update=projection_controls(logits[-1],logits[0],CFG['projection_control_strength'])
        collect('parallel_05',parallel);collect('orthogonal_05',orthogonal)
        for value in CFG['temperature_grid']:collect(f'temperature_{value:g}',logits[-1]/value)
        token_counts.append(len(answer))
        if option==0:
            previous_lp=previous_z=previous_h=None
            # Native gate must use the common prompt endpoint, not final answer.
            endpoint_gates=[gate[:,len(prefix)-1:len(prefix),:] for gate in gates]
            cdf=native_cdf(endpoint_gates)
            for depth,(z,h) in enumerate(zip(logits,selected),1):
                z=z[0];h=h[0,0];stats,lp=vocab_stats(z)
                stats.update(depth=depth,native_gate_cdf=float(cdf[depth-1]),full_vocab_kl=None if previous_lp is None else float((lp.exp()*(lp-previous_lp)).sum()),
                             full_vocab_logit_l2=None if previous_z is None else float((z-previous_z).norm()))
                stats.update(hidden_relative_l2=None,hidden_absolute_l2=None,hidden_cosine=None)
                if previous_h is not None:stats.update(hidden_change(h,previous_h))
                trajectory.append(stats);previous_lp=lp;previous_z=z;previous_h=h
            guided_hidden=model.lm_head(model.model.norm(raw[-1][:,:1]+.5*(raw[-1][:,:1]-raw[0][:,:1])))[0,0].float()
            hidden_update=guided_hidden-logits[-1][0];hidden_update-=hidden_update.mean()
            full_update=logits[-1][0]-logits[0][0];full_update-=full_update.mean()
            geometry=dict(hidden_vs_orthogonal_cosine=float(F.cosine_similarity(hidden_update,orthogonal_update[0],dim=0)),hidden_vs_full_cosine=float(F.cosine_similarity(hidden_update,full_update,dim=0)),
                          parallel_coefficient=float((parallel_update[0]*(logits[-1][0]-logits[-1][0].mean())).sum()/(logits[-1][0]-logits[-1][0].mean()).square().sum().clamp_min(1e-20)))
    predictions={key:candidate_result(values,row['choice_char_lengths'],token_counts,row['gold'],row['primary_metric']) for key,values in sums.items()}
    previous=None
    for depth,state in enumerate(trajectory,1):
        item=predictions['r'+str(depth)];p=np.array(item['choice_weights']);logp=np.array(item['choice_log_weights']);score=np.array(item['scores']);ordered=np.sort(p);ordered_score=np.sort(score)
        state.update(prediction=item['prediction'],correct=item['correct'],choice_confidence=float(p.max()),choice_entropy=float(-(p*logp).sum()),choice_probability_margin=float(ordered[-1]-ordered[-2]),choice_logit_margin=float(ordered_score[-1]-ordered_score[-2]),
                     choice_kl=None if previous is None else float((p*(logp-previous)).sum()),choice_logit_l2=None if previous is None else float(np.linalg.norm(score-previous_score)))
        previous=logp;previous_score=score
    torch.cuda.synchronize()
    return dict(**signature(row),model=name,predictions=predictions,trajectory=trajectory,geometry=geometry,adaptive_strength_min=min(adaptive_strengths),adaptive_strength_max=max(adaptive_strengths),
                elapsed_s=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),note='One shared four-loop forward per candidate; guidance grid readouts share core compute, not independent runtime runs.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=CFG['models']);parser.add_argument('--split',required=True,choices=['pilot','validation','test']);args=parser.parse_args()
    name=args.model;split=args.split
    with Budget('mc_'+name+'_'+split) as budget:
        tok,model=load(name)
        rows=read('data/'+('validation' if split in ['pilot','validation'] else 'qwen_test' if name.startswith('Qwen') else 'ouro_test')+'.jsonl')
        if split=='pilot':
            # Spread critical path coverage across all benchmark families.
            rows=[r for task,n in [('arc_challenge',7),('openbookqa',7),('mmlu',6)] for r in [x for x in rows if x['task']==task or task=='mmlu' and x['task'].startswith('mmlu_')][:n]]
        if name.startswith('Qwen'):
            recipe=Recipe(model)
            for method in CFG['qwen_methods']:
                path=f'trajectories/{name}_{method}_{split}.jsonl';done=existing(path)
                pilot_path=ROOT/f'trajectories/{name}_{method}_pilot.jsonl'
                if split=='validation' and pilot_path.exists():
                    for prior in read(pilot_path):
                        if prior['id'] not in done:
                            reused=dict(prior,pilot_reused=True,pilot_elapsed_s=prior['elapsed_s'],elapsed_s=0)
                            append(path,reused);done[prior['id']]=reused
                recipe.restore()
                if method.startswith('A') and method!='A0':
                    original=json.loads((BASE/'configs/protocol.json').read_text())['qwen_methods'];recipe.set(next(c for c in original if c['id']==method))
                torch.cuda.reset_peak_memory_stats()
                for row in rows:
                    if row['id'] in done:continue
                    budget.check();result=qwen_score(tok,model,recipe,row,method);append(path,result);done[row['id']]=result
                    save('STATUS.json',dict(stage='mc',model=name,split=split,method=method,completed=len(done),total=len(rows)))
                    if len(done)%10==0:print(json.dumps(dict(model=name,split=split,method=method,completed=len(done),total=len(rows))),flush=True)
            recipe.restore()
        else:
            path=f'trajectories/{name}_{split}.jsonl';done=existing(path);torch.cuda.reset_peak_memory_stats()
            pilot_path=ROOT/f'trajectories/{name}_pilot.jsonl'
            if split=='validation' and pilot_path.exists():
                for prior in read(pilot_path):
                    if prior['id'] not in done:
                        reused=dict(prior,pilot_reused=True,pilot_elapsed_s=prior['elapsed_s'],elapsed_s=0)
                        append(path,reused);done[prior['id']]=reused
            for row in rows:
                if row['id'] in done:continue
                budget.check();result=ouro_score(tok,model,row,name);append(path,result);done[row['id']]=result
                save('STATUS.json',dict(stage='mc',model=name,split=split,completed=len(done),total=len(rows)))
                if len(done)%10==0:print(json.dumps(dict(model=name,split=split,completed=len(done),total=len(rows))),flush=True)

if __name__=='__main__':main()
