"""Actual autoregressive fixed/adaptive LoopCD, with true recurrent caches."""
import argparse
from contextlib import nullcontext
from decimal import Decimal,InvalidOperation
import json
import re
import shutil
import time
import numpy as np
import torch
from common import ROOT,CFG,Budget,load,read,save,append,existing,raw_states,vocab_stats,hidden_change
from methods import adaptive_logits

def extract(text):
    matches=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',text)
    return matches[-1].replace(',','').lstrip('+') if matches else None

def numeric_equal(prediction,gold):
    if prediction is None:return False
    try:return Decimal(prediction.replace(',',''))==Decimal(gold.replace(',',''))
    except InvalidOperation:return False

def condition(method,model,selected=None):
    result=dict(method=method,depth=4,kind='baseline',omega=0,cap=0)
    if method.startswith('depth_'):result['depth']=int(method.split('_')[-1])
    elif method=='logits_paper_03':result.update(kind='logits',omega=.3)
    elif method=='logits_adaptive_paper':result.update(kind='adaptive',cap=CFG['loopcd_generation_adaptive_caps'][model])
    elif method=='hidden_fixed_025':result.update(kind='hidden',omega=.25)
    elif method.endswith('_validation_selected'):
        kind=method.split('_')[0];result.update(kind=kind,omega=selected[model][kind]['omega'])
    elif method.startswith(('logits_','hidden_')):
        kind,value=method.split('_');result.update(kind=kind,omega=float(value))
    return result

def canonical(case):
    kind=case['kind'];omega=case['omega'];cap=case['cap']
    if kind in ['logits','hidden'] and omega==0 or kind=='adaptive' and cap==0:kind='baseline'
    return case['depth'],kind,omega if kind!='baseline' else 0,cap if kind=='adaptive' else 0

def set_depth(model,depth):
    model.model.total_ut_steps=depth;model.config.total_ut_steps=depth
    model.early_exit_step=depth-1

def decode(tok,model,prompt,case,budget,max_tokens=None,audit=False):
    max_tokens=max_tokens or CFG['math']['max_new_tokens'];set_depth(model,case['depth'])
    encoded=tok.encode(prompt,add_special_tokens=False)
    assert len(encoded)+max_tokens<=CFG['max_context_tokens']
    input_ids=torch.tensor([encoded],device='cuda');past=None;generated=[];trace=[];strengths=[]
    torch.cuda.synchronize();started=time.monotonic()
    for step in range(max_tokens):
        budget.check(reserve=30)
        capture=raw_states(model,slice(-1,None)) if case['kind']=='hidden' else nullcontext(None)
        with capture as raw:outputs,states,_=model.model(input_ids=input_ids,past_key_values=past,use_cache=True)
        assert len(states)==case['depth']
        if audit:
            logits=[model.lm_head(h[:,-1:,:])[0,0].float() for h in states]
            final=logits[-1][None]
        else:final=model.lm_head(states[-1][:,-1:,:])[:,-1,:].float()
        if case['kind']=='hidden':
            assert len(raw)==case['depth']
            hidden=raw[-1]+case['omega']*(raw[-1]-raw[0])
            guided=model.lm_head(model.model.norm(hidden))[:,-1,:].float();strengths.append(case['omega'])
        elif case['kind'] in ['logits','adaptive']:
            early=model.lm_head(states[0][:,-1:,:])[:,-1,:].float()
            if case['kind']=='adaptive':
                guided,omega=adaptive_logits(final,early,case['cap']);strengths.append(float(omega[0]))
            else:guided=final+case['omega']*(final-early);strengths.append(case['omega'])
        else:guided=final
        if not bool(torch.isfinite(guided).all()):raise FloatingPointError('Nonfinite generation logits: '+str(case))
        token=int(guided.argmax(-1).item());past=outputs.past_key_values
        if step==0:
            layer_count=model.config.num_hidden_layers*case['depth']
            assert all(past.get_seq_length(i)==len(encoded) for i in range(layer_count))
        if audit:
            by_depth=[];previous_lp=previous_h=None
            for depth,(z,state) in enumerate(zip(logits,states),1):
                stats,lp=vocab_stats(z);h=state[0,-1,:]
                stats.update(depth=depth,prediction=int(z.argmax()),full_vocab_kl=None if previous_lp is None else float((lp.exp()*(lp-previous_lp)).sum()),hidden_relative_l2=None,hidden_absolute_l2=None,hidden_cosine=None)
                if previous_h is not None:stats.update(hidden_change(h,previous_h))
                by_depth.append(stats);previous_lp=lp;previous_h=h
            trace.append(dict(position=step,chosen_token_id=token,by_depth=by_depth,scope='same generated prefix; token agreement is not token correctness'))
        if token==tok.eos_token_id:break
        generated.append(token);input_ids=torch.tensor([[token]],device='cuda')
        text=tok.decode(generated,skip_special_tokens=True)
        if '\nQuestion:' in text or '\nQ:' in text:break
        if re.search(r'####\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\n',text):break
    torch.cuda.synchronize()
    text=tok.decode(generated,skip_special_tokens=True).split('\nQuestion:')[0].split('\nQ:')[0]
    return dict(text=text,prediction=extract(text),token_ids=generated,new_tokens=len(generated),reached_token_limit=len(generated)==max_tokens,
                elapsed_s=time.monotonic()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),
                cache_layers=model.config.num_hidden_layers*case['depth'],token_trajectory=trace,
                baseline_audit_extra_heads=audit,actual_strength_min=min(strengths) if strengths else 0,actual_strength_max=max(strengths) if strengths else 0)

def generation_parity(tok,model,budget,name):
    prompt='The Solar System contains planets and';reports=[]
    for depth in [2,3,4]:
        set_depth(model,depth);x=tok(prompt,return_tensors='pt').to('cuda')
        output=model.generate(**x,max_new_tokens=8,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,return_dict_in_generate=True)
        expected=output.sequences[0,x.input_ids.shape[1]:].tolist()
        if tok.eos_token_id in expected:expected=expected[:expected.index(tok.eos_token_id)]
        case=dict(method='parity',depth=depth,kind='baseline',omega=0,cap=0)
        actual=decode(tok,model,prompt,case,budget,max_tokens=8,audit=False)
        assert actual['token_ids']==expected,(depth,actual['token_ids'],expected)
        if depth==4:
            for kind in ['logits','hidden','adaptive']:
                zero=dict(case,kind=kind)
                result=decode(tok,model,prompt,zero,budget,max_tokens=8,audit=False)
                assert result['token_ids']==expected,(kind,result['token_ids'],expected)
            audited=decode(tok,model,prompt,case,budget,max_tokens=8,audit=True)
            assert audited['token_ids']==expected
        reports.append(dict(depth=depth,tokens=expected,cache_layers=actual['cache_layers'],passed=True))
    set_depth(model,4);save('results/generation_parity_'+name+'.json',dict(passed=True,model=name,cases=reports,scope='HF.generate, true-depth 2/3/4 caching and zero guidance; baseline scalar tracing preserves tokens'))

def run_cases(tok,model,budget,name,rows,methods,destination,shots,selected=None,phase='validation'):
    done=existing(destination,key='record_key')
    for row in rows:
        available={}
        for old in done.values():
            if old['id']==row['id']:available[canonical(old['condition'])]=old
        prompt=shots+'\n\nQuestion: '+row['question']+'\nAnswer:'
        for method in methods:
            key=row['id']+'|'+method
            if key in done:continue
            budget.check()
            if shutil.disk_usage(ROOT).free<CFG['free_disk_reserve_gib']*1024**3:raise RuntimeError('Disk free reserve reached')
            case=condition(method,name,selected);same=available.get(canonical(case))
            if same is not None:
                result={**same,'record_key':key,'method':method,'condition':case,'identity_reused_from':same['method'],'elapsed_s':0}
            else:
                outcome=decode(tok,model,prompt,case,budget,audit=method=='baseline' and case['depth']==4 and CFG['generation_audit']['baseline_only_all_depth_token_scalars'])
                result=dict(id=row['id'],task='gsm8k',model=name,method=method,record_key=key,gold=row['gold'],condition=case,phase=phase,
                    max_new_tokens=CFG['math']['max_new_tokens'],**outcome)
                result['correct']=result['prediction']==row['gold'].replace(',','')
                result['numeric_correct']=numeric_equal(result['prediction'],row['gold'])
            append(destination,result);done[key]=result;available[canonical(case)]=result
            save('STATUS.json',dict(stage='generation',phase=phase,model=name,method=method,completed=len(done),expected=len(rows)*len(methods)))
            if len(done)%5==0:print(json.dumps(dict(model=name,phase=phase,method=method,completed=len(done),expected=len(rows)*len(methods))),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['Ouro-1.4B','Ouro-2.6B'],required=True);parser.add_argument('--stage',choices=['parity','pilot','validation','test'],required=True);args=parser.parse_args()
    with Budget('generation_'+args.model+'_'+args.stage) as budget:
        tok,model=load(args.model)
        if args.stage=='parity':generation_parity(tok,model,budget,args.model);return
        assert json.loads((ROOT/'results'/('generation_parity_'+args.model+'.json')).read_text())['passed']
        examples=read('data/math_fewshot.jsonl');shots='\n\n'.join('Question: '+row['question']+'\nAnswer: '+row['answer'] for row in examples)
        if args.stage in ['pilot','validation']:
            rows=read('data/math_validation.jsonl')
            if args.stage=='pilot':rows=rows[:5]
            run_cases(tok,model,budget,args.model,rows,CFG['math']['validation_methods'],f'trajectories/math_{args.model}_validation.jsonl',shots,phase='validation')
            if args.stage=='pilot':
                run_cases(tok,model,budget,args.model,rows,['depth_2','depth_3','logits_paper_03','logits_adaptive_paper'],f'trajectories/math_{args.model}_pilot_extra.jsonl',shots,phase='pilot_extra')
        else:
            selected=json.loads((ROOT/'configs/math_selected.json').read_text());rows=read('data/math_test.jsonl')
            execution=ROOT/'configs/math_execution.json'
            if execution.exists():rows=rows[:json.loads(execution.read_text())['test_count']]
            run_cases(tok,model,budget,args.model,rows,CFG['math']['test_methods'],f'trajectories/math_{args.model}_test.jsonl',shots,selected,phase='test')
        set_depth(model,4)

if __name__=='__main__':main()
