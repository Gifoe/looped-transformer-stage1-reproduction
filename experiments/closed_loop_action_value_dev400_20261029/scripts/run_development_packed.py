"""Counterfactual recurrent state actions on a fixed exposed development set."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT.parent/'closed_loop_action_value_20261029'
OLD=ROOT.parent/'frozen_loop_update_value_seed2021_v1'
sys.path.insert(0,str(PILOT/'scripts'))
import pilot as frozen
import torch
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM
ALPHAS=frozen.ALPHAS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(ROOT/'results'/name).write_text(json.dumps(x,indent=2),encoding='utf8')
def sync():torch.cuda.synchronize()

@torch.no_grad()
def decision_with_controls(tok,model,prompt,question):
    captured=[]
    handle=model.model.norm.register_forward_hook(lambda mod,args,out:captured.append(out.detach()))
    sync();start=time.monotonic()
    try:decision=frozen.choose(tok,model,prompt,question)
    finally:handle.remove()
    # Insufficient probes skip the pilot's forward pass; get the same pre-answer
    # states solely for the predefined confidence control in that case.
    if not captured:
        old=model.model.total_ut_steps
        try:
            model.model.total_ut_steps=2
            _,captured,_=model.model(input_ids=torch.tensor([tok.encode(prompt,add_special_tokens=False)],device='cuda'),use_cache=False)
        finally:model.model.total_ut_steps=old
    assert len(captured)==2
    h1,h2=captured;controls={}
    for a in ALPHAS:
        h=h2 if a==0 else (h2.float()+a*(h2.float()-h1.float())).to(h2.dtype)
        lp=model.lm_head(h[:,-1,:])[0].float().log_softmax(-1);p=lp.exp();top=p.topk(2).values
        controls[str(a)]=dict(margin=float(top[0]-top[1]),entropy=float(-(p*lp).sum()))
    decision['confidence_alpha']=max(ALPHAS,key=lambda a:controls[str(a)]['margin'])
    sync();return decision,controls,time.monotonic()-start

@torch.no_grad()
def evaluate(tok,model,pairs,alpha,question):
    sums=[];terminal={str(a):[] for a in ALPHAS};trace=None;effects=[];terminal_seconds=0
    sync();start=time.monotonic()
    for ci,(prompt,answer) in enumerate(pairs):
        p,t=frozen.encode(tok,prompt,answer);x=torch.tensor([p+t[:-1]],device='cuda')
        hook=frozen.Intervention(alpha);handle=model.model.norm.register_forward_hook(hook)
        try:_,states,_=model.model(input_ids=x,use_cache=False)
        finally:handle.remove()
        assert hook.count==4
        pi=frozen.probes(tok,prompt,question,p) if ci==0 else [];npb=len(pi)
        positions=[i-1 for i in pi]+list(range(len(p)-1,len(p)+len(t)-1));target=torch.tensor([p[i] for i in pi]+t,device='cuda')[:,None]
        z1=model.lm_head(states[0][:,positions,:])[0].float();z4=model.lm_head(states[-1][:,positions,:])[0].float()
        def ll(last):
            top=last.softmax(-1).topk(2,dim=-1).values;w=1-(top[:,0]-top[:,1]);z=last+w[:,None]*(last-z1)
            return float(z.log_softmax(-1).gather(-1,target)[npb:].sum())
        sums.append(ll(z4));effects.append(hook.effect)
        if alpha==0:
            terminal['0.0'].append(sums[-1]);sync();t0=time.monotonic()
            d=states[1].float()-states[0].float()
            for a in ALPHAS[1:]:
                h=(states[-1].float()+a*d).to(states[-1].dtype)
                terminal[str(a)].append(ll(model.lm_head(h[:,positions,:])[0].float()))
            sync();terminal_seconds+=time.monotonic()-t0
        if ci==0:trace=torch.stack([h[0,len(p)-1,:].float() for h in states]).cpu()
    sync();return sums,terminal,effects,trace,dict(total_seconds=time.monotonic()-start,terminal_diagnostic_seconds=terminal_seconds)

def main():
    begin=time.monotonic();deadline=begin+3500
    lock=json.loads((ROOT/'configs/LOCK.json').read_text())
    for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,rel
    amendment=json.loads((ROOT/'configs/PACKING_AMENDMENT.json').read_text())
    for rel,h in amendment['new_files'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in lock['dependencies'].items():assert sha(ROOT.parent/rel)==h,rel
    with (OLD/'data/discovery.jsonl').open(encoding='utf8') as f:rows=[json.loads(l) for l in f]
    assert len(rows)==400 and [r['id'] for r in rows]==lock['ids']
    with (OLD/'results/discovery.jsonl').open(encoding='utf8') as f:reference={r['id']:r['views']['official']['sum_scores']['adaptive'] for r in map(json.loads,f)}
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
    directory=ROOT.parent/'looped_stage1_trajectory_audit_seed2021_v1/models/Ouro-1.4B'
    meta=json.loads((directory/'snapshot_metadata.json').read_text());assert meta['sha']=='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
    tok=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    model,info=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.to('cuda').eval();model.requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
    assert model.model.total_ut_steps==4
    torch.cuda.reset_peak_memory_stats();maxerr=0.;records=[]
    with (ROOT/'results/development_packed.jsonl').open('x',encoding='utf8') as out:
        for row in rows:
            assert time.monotonic()<deadline,'Budget reserve'
            start=time.monotonic();pairs=row['variants']['official'];assert len({p[0] for p in pairs})==1
            decision,controls,decision_s=decision_with_controls(tok,model,pairs[0][0],row['question'])
            actions={};traces={};terminal=None
            for a in ALPHAS:
                scores,term,effects,trace,timing=evaluate(tok,model,pairs,a,row['question']);traces[a]=trace
                actions[str(a)]=dict(sum_scores=scores,prediction=int(np.argmax(np.array(scores)/row['choice_char_lengths'])),intervention_rms=effects,**timing)
                if a==0:terminal={k:dict(sum_scores=v,prediction=int(np.argmax(np.array(v)/row['choice_char_lengths']))) for k,v in term.items()}
            drift={str(a):[float((traces[a][i]-traces[0.][i]).square().mean().sqrt()) for i in range(4)] for a in ALPHAS[1:]}
            assert all(v[0]==0 and min(v[1:])>0 for v in drift.values())
            error=max(abs(x-y) for x,y in zip(actions['0.0']['sum_scores'],reference[row['id']]))
            assert error<1e-5,error
            maxerr=max(maxerr,error)
            # No gold is supplied to the controller or any forward pass.
            rec=dict(id=row['id'],decision=decision,controls=controls,actions=actions,terminal=terminal,downstream_state_rms=drift,baseline_score_error=error,decision_seconds=decision_s,choice_char_lengths=row['choice_char_lengths'],gold=row['gold'],elapsed_s=time.monotonic()-start)
            out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush();records.append(rec)
            if len(records)%10==0:print(json.dumps(dict(completed=len(records),expected=400,elapsed_s=time.monotonic()-begin)),flush=True)
    assert all(p._version==versions[n] and not p.requires_grad for n,p in model.named_parameters())
    save('COMPLETE_PACKED.json',dict(n=len(records),weights_unchanged=True,baseline_max_score_error=maxerr,all_nonzero_actions_propagated=True,elapsed_s=time.monotonic()-begin,peak_vram_bytes=torch.cuda.max_memory_allocated(),sha256=sha(ROOT/'results/development_packed.jsonl'),final_tasks_loaded=False))

if __name__=='__main__':main()
