"""Audit the pinned official Ouro implementation without changing recurrence.

Each readout uses the same candidate prefix. Hidden guidance is applied before
the output norm; returned already-normalized states never receive a second norm.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import re
import time

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CFG=json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf-8-sig'))
random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False

def save(path,obj):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    for attempt in range(40):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)

class Budget:
    def __init__(self):
        self.path=ROOT/'logs/gpu_budget.json'
        self.previous=json.loads(self.path.read_text()) if self.path.exists() else dict(used_seconds=0,runs=[])
        if not self.previous.get('environment_smoke_reserved_seconds'):
            # The tiny preparatory BF16 matrix smoke was outside the evaluator.
            # Charge a conservative minute, not an invented precise duration.
            self.previous['used_seconds']+=60
            self.previous['environment_smoke_reserved_seconds']=60
        self.start=time.monotonic()
    def check(self,reserve=60):
        used=self.previous['used_seconds']+time.monotonic()-self.start
        if used+reserve>=CFG['hard_gpu_hours']*3600:raise RuntimeError('50 GPU-hour hard cap reached')
    def close(self,tag):
        elapsed=time.monotonic()-self.start
        self.previous['used_seconds']+=elapsed
        self.previous['runs'].append(dict(tag=tag,seconds=elapsed))
        save('logs/gpu_budget.json',self.previous)

def load():
    p=ROOT/'models/Ouro-1.4B'
    tok=AutoTokenizer.from_pretrained(p,local_files_only=True)
    m=AutoModelForCausalLM.from_pretrained(p,trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    assert m.config.num_hidden_layers==len(m.model.layers)
    assert m.model.total_ut_steps==4
    return tok,m

def discrepancy(a,b):
    delta=(a.float()-b.float()).abs()
    return dict(max_abs=delta.max().item(),relative_l2=(delta.norm()/a.float().norm().clamp_min(1e-8)).item(),argmax_agreement=(a.argmax(-1)==b.argmax(-1)).float().mean().item())

@contextmanager
def capture_raw(m,selection):
    states=[]
    handle=m.model.norm.register_forward_pre_hook(lambda module,args:states.append(args[0][:,selection,:].clone()))
    try:yield states
    finally:handle.remove()

def parity(tok,m):
    reports=[]
    for text in ['Question: What is 2 + 2?\nAnswer:', 'Water freezes at zero degrees Celsius.', 'Question: A fair coin is tossed twice.\nAnswer:']:
        x=tok(text,return_tensors='pt').to('cuda')
        normal=m(**x,use_cache=False,logits_to_keep=1).logits
        with capture_raw(m,slice(-1,None)) as raw:
            outputs,states,gates=m.model(**x,use_cache=False)
        assert len(states)==len(raw)==4
        direct=m.lm_head(states[-1][:,-1:,:])
        via_raw=m.lm_head(m.model.norm(raw[-1]))
        reports.append(dict(default_vs_extracted=discrepancy(normal,direct),raw_norm_vs_returned=discrepancy(direct,via_raw)))
        assert torch.equal(normal,direct),reports[-1]
        assert torch.equal(direct,via_raw),reports[-1]
        # Confirm the public readout selector uses 0-based depth and same states.
        for depth in range(4):
            z=m(**x,use_cache=False,exit_at_step=depth,logits_to_keep=1).logits
            assert torch.equal(z,m.lm_head(states[depth][:,-1:,:]))
        # Real incremental decoding must match no-cache scoring of the prefix.
        cached=m(**x,use_cache=True,logits_to_keep=1)
        token=cached.logits[:,-1,:].argmax(-1,keepdim=True)
        incremental=m(input_ids=token,past_key_values=cached.past_key_values,use_cache=True,logits_to_keep=1).logits
        whole=m(input_ids=torch.cat([x.input_ids,token],dim=1),use_cache=False,logits_to_keep=1).logits
        cache_diff=discrepancy(whole,incremental)
        reports[-1]['cache_vs_full']=cache_diff
        assert cache_diff['relative_l2']<0.02 and cache_diff['argmax_agreement']==1.0,cache_diff
    report=dict(passed=True,cases=reports,config=m.config.to_dict(),
                model_sha=json.loads((ROOT/'models/Ouro-1.4B/snapshot_metadata.json').read_text())['sha'],
                source_sha256=hashlib.sha256((ROOT/'models/Ouro-1.4B/modeling_ouro.py').read_bytes()).hexdigest(),
                torch=torch.__version__,transformers=metadata.version('transformers'),peak_memory_bytes=torch.cuda.max_memory_allocated())
    save('results/ouro_parity.json',report)
    return report

def prompt_for(row):
    if row['task'].startswith('mmlu_'):
        subject=row['task'].removeprefix('mmlu_').replace('_',' ')
        choices='\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(row['choices']))
        return f'The following are multiple choice questions (with answers) about {subject}.\n\n'+row['question']+'\n'+choices+'\nAnswer:', [' '+chr(65+i) for i in range(len(row['choices']))]
    return 'Question: '+row['question']+'\nAnswer:', [' '+x for x in row['choices']]

def encode_pair(tok,prompt,answer):
    # Explicit boundary convention, fixed across all methods. No implicit BOS.
    prefix=tok.encode(prompt,add_special_tokens=False)
    full=tok.encode(prompt+answer,add_special_tokens=False)
    if full[:len(prefix)]!=prefix:
        raise RuntimeError('Prompt/answer token boundary is not prefix-consistent')
    continuation=full[len(prefix):]
    assert continuation
    if len(full)>CFG['scoring']['prompt_max_tokens']:raise RuntimeError('Context limit exceeded; no truncation permitted')
    return prefix,continuation

def distribution(z):
    logp=z.float().log_softmax(-1);p=logp.exp();v=p.topk(2).values
    return dict(confidence=v[0].item(),entropy=-(p*logp).sum().item(),margin=(v[0]-v[1]).item()),logp

def run_sample(tok,m,row):
    prompt,answers=prompt_for(row)
    sums={f'r{r}':[] for r in range(1,5)}
    for omega in CFG['loopcd_strength_grid']:
        sums['logits_'+str(omega)]=[];sums['hidden_'+str(omega)]=[]
    for temperature in CFG['temperature_grid']:sums['temp_'+str(temperature)]=[]
    token_lengths=[];char_lengths=[];trajectory=[]
    torch.cuda.synchronize();start=time.monotonic()
    for option,answer in enumerate(answers):
        prefix,continuation=encode_pair(tok,prompt,answer)
        tokens=torch.tensor([prefix+continuation[:-1]],device='cuda')
        selection=slice(len(prefix)-1,None)
        target=torch.tensor(continuation,device='cuda')
        with capture_raw(m,selection) as raw:
            _,states,_=m.model(input_ids=tokens,use_cache=False)
        selected=[h[:,selection,:] for h in states]
        logits=[m.lm_head(h)[0].float() for h in selected]
        for depth,z in enumerate(logits):
            sums['r'+str(depth+1)].append(z.log_softmax(-1).gather(-1,target[:,None]).sum().item())
        for omega in CFG['loopcd_strength_grid']:
            guided=logits[-1]+omega*(logits[-1]-logits[0])
            sums['logits_'+str(omega)].append(guided.log_softmax(-1).gather(-1,target[:,None]).sum().item())
            # BF16 hidden combination, exactly one output norm and one head.
            guided_h=raw[-1]+omega*(raw[-1]-raw[0])
            zh=m.lm_head(m.model.norm(guided_h))[0].float()
            sums['hidden_'+str(omega)].append(zh.log_softmax(-1).gather(-1,target[:,None]).sum().item())
        for temperature in CFG['temperature_grid']:
            sums['temp_'+str(temperature)].append((logits[-1]/temperature).log_softmax(-1).gather(-1,target[:,None]).sum().item())
        if option==0:
            prev_logp=prev_h=None
            for depth,(z,h) in enumerate(zip(logits,selected)):
                stats,logp=distribution(z[0]);h=h[0,0,:].float()
                stats.update(depth=depth+1,full_vocab_kl=None if prev_logp is None else (logp.exp()*(logp-prev_logp)).sum().item(),
                             hidden_relative_l2=None if prev_h is None else ((h-prev_h).norm()/prev_h.norm().clamp_min(1e-8)).item(),
                             hidden_cosine=None if prev_h is None else F.cosine_similarity(h,prev_h,dim=0).item())
                trajectory.append(stats);prev_logp=logp;prev_h=h
        token_lengths.append(len(continuation));char_lengths.append(len(answer.lstrip(' ')))
        del states,selected,raw,logits
    predictions={}
    for key,values in sums.items():
        scores=np.asarray(values)/np.maximum(char_lengths,1)
        prob=np.exp(scores-scores.max());prob/=prob.sum()
        pred=int(scores.argmax())
        predictions[key]=dict(prediction=pred,correct=pred==row['gold'],scores=scores.tolist(),sum_loglikelihood=values,
                               choice_probabilities=prob.tolist(),gold_choice_logp=float(np.log(prob[row['gold']])),
                               gold_nll_per_token=-values[row['gold']]/token_lengths[row['gold']])
    previous=None
    for depth,stats in enumerate(trajectory):
        item=predictions['r'+str(depth+1)];p=np.asarray(item['choice_probabilities'])
        order=np.sort(p)
        stats.update(prediction=item['prediction'],correct=item['correct'],gold_choice_logp=item['gold_choice_logp'],
                     choice_confidence=float(p.max()),choice_entropy=float(-(p*np.log(p)).sum()),choice_margin=float(order[-1]-order[-2]),
                     choice_kl=None if previous is None else float((p*(np.log(p)-np.log(previous))).sum()))
        previous=p
    torch.cuda.synchronize()
    return dict(**row,model='ByteDance/Ouro-1.4B',protocol='zero_shot_complementary_audit_v1',prompt=prompt,
                prompt_tokens=len(prefix),token_lengths=token_lengths,char_lengths=char_lengths,
                predictions=predictions,trajectory=trajectory,elapsed_s=time.monotonic()-start,
                peak_memory_bytes=torch.cuda.max_memory_allocated())

def benchmark(tok,m,rows,budget):
    records=[]
    for r in [1,2,3,4]:
        m.model.total_ut_steps=r;m.config.total_ut_steps=r
        torch.cuda.reset_peak_memory_stats()
        for i,row in enumerate(rows[:20]):
            budget.check()
            prompt,_=prompt_for(row);x=tok(prompt,return_tensors='pt').to('cuda')
            torch.cuda.synchronize();start=time.monotonic()
            outputs,states,_=m.model(**x,use_cache=False)
            z=m.lm_head(states[-1][:,-1:,:]);torch.cuda.synchronize()
            records.append(dict(depth=r,n_tokens=x.input_ids.shape[1],elapsed_s=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated()))
    m.model.total_ut_steps=4;m.config.total_ut_steps=4
    # End-to-end diagnostic cost includes all candidate readouts and LoopCD grid.
    pilots=[]
    for row in rows[:3]:budget.check();pilots.append(run_sample(tok,m,row))
    mean=sum(x['elapsed_s'] for x in pilots)/len(pilots)
    estimate=dict(core_choice_samples=400,seconds_per_sample=mean,choice_gpu_hours=400*mean/3600,
                  generation_reserve_gpu_hours=5,other_reserve_gpu_hours=3)
    estimate['estimated_total_gpu_hours']=estimate['choice_gpu_hours']+8
    report=dict(records=records,pilot_times=[x['elapsed_s'] for x in pilots],estimate=estimate,
                passes_budget=estimate['estimated_total_gpu_hours']<45,
                note='Depth timing measures real truncated forwards; formal trajectory extraction still executes all four depths.')
    save('results/performance_preflight.json',report)
    assert report['passes_budget'],'Budget estimate too large; reduce immutable manifest before formal evaluation'
    return report

def evaluate(tok,m,manifest,dest,budget,limit=None):
    rows=[json.loads(l) for l in (ROOT/'data'/manifest).read_text(encoding='utf-8').splitlines()]
    if limit:rows=rows[:limit]
    dest=ROOT/'trajectories'/dest
    existing=[json.loads(l) for l in dest.read_text(encoding='utf-8').splitlines()] if dest.exists() else []
    assert len({r['id'] for r in existing})==len(existing)
    done={r['id'] for r in existing}
    with dest.open('a',encoding='utf-8') as f:
        for row in rows:
            if row['id'] in done:continue
            budget.check();result=run_sample(tok,m,row)
            f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();done.add(row['id'])
            save('STATUS.json',dict(stage='ouro_evaluation',manifest=manifest,completed=len(done),total=len(rows)))
            print(json.dumps(dict(id=row['id'],done=len(done),n=len(rows),elapsed_s=result['elapsed_s'])),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['parity','benchmark','pilot','formal'],required=True);args=parser.parse_args()
    budget=Budget()
    try:
        budget.check();tok,m=load()
        if args.stage=='parity':parity(tok,m)
        else:
            assert json.loads((ROOT/'results/ouro_parity.json').read_text())['passed']
            rows=[json.loads(l) for l in (ROOT/'data/validation.jsonl').read_text(encoding='utf-8').splitlines()]
            if args.stage=='benchmark':benchmark(tok,m,rows,budget)
            elif args.stage=='pilot':evaluate(tok,m,'validation.jsonl','ouro_validation.jsonl',budget,limit=50)
            else:
                assert json.loads((ROOT/'results/performance_preflight.json').read_text())['passes_budget']
                pilot=ROOT/'trajectories/ouro_validation.jsonl';assert pilot.exists() and len(pilot.read_text().splitlines())>=50
                evaluate(tok,m,'validation.jsonl','ouro_validation.jsonl',budget)
                evaluate(tok,m,'ouro_test.jsonl','ouro_test.jsonl',budget)
    finally:budget.close(args.stage)

if __name__=='__main__':main()
