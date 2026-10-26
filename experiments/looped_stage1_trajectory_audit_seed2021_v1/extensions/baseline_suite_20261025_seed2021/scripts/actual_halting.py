"""Actual current-prefix early stop in published Ouro forward; no cache."""
import argparse
from contextlib import contextmanager
import json
import time
import numpy as np
import torch
from common import ROOT,CFG,Budget,load,read,save,append,existing,encode_pair,candidate_result,hidden_change
from policies import exits,exit_index

CATEGORIES=['arc_challenge','openbookqa','mmlu']
category=lambda row:'mmlu' if row['task'].startswith('mmlu_') else row['task']

class StopRecurrentForward(Exception):pass

def set_depth(model,depth):
    model.model.total_ut_steps=depth;model.config.total_ut_steps=depth;model.early_exit_step=depth-1

@contextmanager
def layer_counter(model):
    count=[0]
    def hook(module,args,result):count[0]+=1
    handles=[layer.register_forward_hook(hook) for layer in model.model.layers]
    try:yield count
    finally:
        for handle in handles:handle.remove()

def adaptive_first_candidate(model,x,selection,metric,threshold):
    """Decisions see current/history only; no gold or future state is passed."""
    depth=0;previous=None;last=None;last_z=None;pdf=[];remaining=None;history=[]
    def norm_hook(module,args,result):
        nonlocal depth,previous,last,last_z
        depth+=1;last=result[:,selection,:];last_z=None
        if metric=='native_gate_cdf':return
        if metric=='full_vocab_kl':
            last_z=model.lm_head(last)[0].float();lp=last_z[0].log_softmax(-1)
            value=None if previous is None else float((lp.exp()*(lp-previous)).sum());previous=lp
        elif metric=='hidden_relative_l2':
            h=last[0,0];value=None if previous is None else hidden_change(h,previous)['hidden_relative_l2'];previous=h
        else:raise ValueError(metric)
        history.append(dict(depth=depth,metric=metric,value=value))
        if depth==4 or exits(value,metric,threshold):raise StopRecurrentForward()
    def gate_hook(module,args,result):
        nonlocal remaining
        if metric!='native_gate_cdf':return
        gate=result[0,selection,0][0]
        if remaining is None:remaining=torch.ones_like(gate)
        probability=torch.sigmoid(gate)
        if depth<4:pdf.append(probability*remaining);remaining=remaining*(1-probability)
        else:pdf.append(remaining)
        value=float(torch.stack(pdf).cumsum(0)[-1]);history.append(dict(depth=depth,metric=metric,value=value))
        if depth==4 or exits(value,metric,threshold):raise StopRecurrentForward()
    handles=[model.model.norm.register_forward_hook(norm_hook),model.model.early_exit_gate.register_forward_hook(gate_hook)]
    try:
        try:model.model(input_ids=x,use_cache=False)
        except StopRecurrentForward:pass
        else:raise AssertionError('Stop hook did not run')
    finally:
        for handle in handles:handle.remove()
    assert depth in [1,2,3,4]
    z=last_z if last_z is not None else model.lm_head(last)[0].float()
    return z,depth,history

def prepare(tok,row):
    pairs=[]
    for prompt,continuation in row['request_pairs']:
        prefix,answer=encode_pair(tok,prompt,continuation)
        pairs.append((torch.tensor([prefix+answer[:-1]],device='cuda'),torch.tensor(answer,device='cuda'),slice(len(prefix)-1,None)))
    return pairs

def score(model,pairs,method,threshold=None,verify_layers=False):
    set_depth(model,4);sums=[];counts=[];depth=None;history=[]
    def execute():
        nonlocal depth,history
        for option,(x,target,selection) in enumerate(pairs):
            if option==0 and not method.startswith('fixed'):
                z,depth,history=adaptive_first_candidate(model,x,selection,method,threshold)
            else:
                if option==0:depth=int(method[-1])
                set_depth(model,depth)
                _,states,_=model.model(input_ids=x,use_cache=False)
                assert len(states)==depth
                z=model.lm_head(states[-1][:,selection,:])[0].float()
            sums.append(float(z.log_softmax(-1).gather(-1,target[:,None]).sum()));counts.append(len(target))
    try:
        torch.cuda.synchronize();started=time.monotonic()
        if verify_layers:
            with layer_counter(model) as counter:execute()
            actual_layers=counter[0];assert actual_layers==len(pairs)*len(model.model.layers)*depth,(actual_layers,depth)
        else:execute();actual_layers=None
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        return dict(sum_scores=sums,token_counts=counts,depth=depth,history=history,elapsed_s=elapsed,actual_decoder_layer_calls=actual_layers,expected_decoder_layer_calls=len(pairs)*len(model.model.layers)*depth)
    finally:set_depth(model,4)

def policy_settings(model):
    selected=json.loads((ROOT/'configs/halting_selected.json').read_text())[model];result={}
    for metric in ['full_vocab_kl','hidden_relative_l2','native_gate_cdf']:
        result[metric]={}
        for task in CATEGORIES:
            choices=selected[metric][task]
            target=next((x for x in [CFG['halting']['actual_exit_validation_target'],*CFG['actual_exit_fallback_validation_targets']] if str(x) in choices),None)
            assert target is not None,(metric,task,'No feasible validation policy')
            result[metric][task]=dict(target=target,threshold=choices[str(target)])
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=['Ouro-1.4B','Ouro-2.6B']);args=parser.parse_args();name=args.model
    settings=policy_settings(name);save('configs/actual_halting_'+name+'.json',settings)
    all_rows=read('data/ouro_test.jsonl');rows=[row for task in CATEGORIES for row in [r for r in all_rows if category(r)==task][:10]]
    assert len(rows)==CFG['halting']['actual_latency_questions']
    offline={r['id']:r for r in read(f'trajectories/{name}_test.jsonl')}
    destination=f'trajectories/actual_halting_{name}.jsonl';done=existing(destination,key='record_key')
    methods=['fixed4','fixed2',*settings];parity=[]
    with Budget('actual_halting_'+name) as budget:
        tok,model=load(name);encoded={r['id']:prepare(tok,r) for r in rows}
        # Warm every changed policy path on each benchmark; exclude from timing.
        for task in CATEGORIES:
            row=next(r for r in rows if category(r)==task)
            for method in methods:score(model,encoded[row['id']],method,None if method.startswith('fixed') else settings[method][task]['threshold'])
        for row in rows:
            reference=offline[row['id']];task=category(row)
            for method in methods:
                budget.check();threshold=None if method.startswith('fixed') else settings[method][task]['threshold']
                result=score(model,encoded[row['id']],method,threshold,verify_layers=True)
                expected=int(method[-1]) if method.startswith('fixed') else exit_index(reference['trajectory'],method,threshold)+1
                assert result['depth']==expected,(row['id'],method,result['depth'],expected,result['history'])
                expected_scores=reference['predictions'][f'r{expected}']['sum_scores']
                error=max(abs(a-b) for a,b in zip(result['sum_scores'],expected_scores));assert error<.02,(row['id'],method,error)
                scored=candidate_result(result['sum_scores'],row['choice_char_lengths'],result['token_counts'],row['gold'],row['primary_metric'])
                expected_result=reference['predictions'][f'r{expected}']
                assert all(scored[k]==expected_result[k] for k in ['prediction','raw_prediction','normalized_prediction'])
                parity.append(dict(id=row['id'],method=method,depth=expected,max_score_error=error,actual_decoder_layer_calls=result['actual_decoder_layer_calls'],passed=True))
        save('results/actual_halting_parity_'+name+'.json',dict(passed=True,cases=parity,scope='All 30 questions x five methods; actual layer calls, exit and likelihood parity with full-depth source states'))
        rng=np.random.default_rng(CFG['seed'])
        for repeat in range(CFG['halting']['actual_latency_repeats']):
            for position in rng.permutation(len(rows)):
                row=rows[int(position)];task=category(row)
                for method in rng.permutation(methods).tolist():
                    key=f'{row["id"]}|{method}|{repeat}'
                    if key in done:continue
                    budget.check();threshold=None if method.startswith('fixed') else settings[method][task]['threshold']
                    torch.cuda.reset_peak_memory_stats();result=score(model,encoded[row['id']],method,threshold)
                    scored=candidate_result(result['sum_scores'],row['choice_char_lengths'],result['token_counts'],row['gold'],row['primary_metric'])
                    expected=int(method[-1]) if method.startswith('fixed') else exit_index(offline[row['id']]['trajectory'],method,threshold)+1
                    assert result['depth']==expected
                    reference=offline[row['id']]['predictions'][f'r{expected}']
                    assert all(scored[k]==reference[k] for k in ['prediction','raw_prediction','normalized_prediction'])
                    score_error=max(abs(a-b) for a,b in zip(result['sum_scores'],reference['sum_scores']));assert score_error<.02
                    record=dict(record_key=key,id=row['id'],task=row['task'],model=name,method=method,repeat=repeat,threshold=threshold,**result,result=scored,measured_repeat_max_score_error=score_error,measured_repeat_quality_matches_offline=True,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Pretokenized complete multiple-choice likelihood scoring, use_cache=False; first-candidate online decision then same actual depth for other candidates')
                    append(destination,record);done[key]=record
                    save('STATUS.json',dict(stage='actual_halting',model=name,completed=len(done),expected=len(rows)*len(methods)*CFG['halting']['actual_latency_repeats']))
                if len(done)%50==0:print(json.dumps(dict(model=name,completed=len(done))),flush=True)
    assert len(done)==750

if __name__=='__main__':main()
