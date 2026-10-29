"""Single precommitted state-feedback implementation pilot, not performance evidence."""
from pathlib import Path
import sys, json, time, hashlib, os
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'frozen_loop_update_value_seed2021_v1'
sys.path.insert(0,str(OLD/'scripts'))
import runtime_bootstrap
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from score import encode, probes, action_logits

ALPHAS=[0.,-1.,-.5,.5]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
    (ROOT/'results'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

class Intervention:
    def __init__(self,alpha):self.alpha=alpha;self.count=0;self.first=None;self.effect=None
    def __call__(self,module,args,out):
        self.count+=1
        if self.count==1:self.first=out.detach().clone()
        if self.count==2:
            y=out if self.alpha==0 else (out.float()+self.alpha*(out.float()-self.first.float())).to(out.dtype)
            self.effect=float((y.float()-out.float()).square().mean().sqrt())
            return y
        return out

@torch.no_grad()
def choose(tok,model,prompt,question):
    prefix=tok.encode(prompt,add_special_tokens=False)
    pi=probes(tok,prompt,question,prefix)
    if len(pi)<4:return dict(alpha=0.,reason='insufficient_known_input_probes',token_indices=pi)
    old=model.model.total_ut_steps
    try:
        model.model.total_ut_steps=2
        _,states,_=model.model(input_ids=torch.tensor([prefix],device='cuda'),use_cache=False)
    finally:model.model.total_ut_steps=old
    h1,h2=states;idx=[i-1 for i in pi];target=torch.tensor([prefix[i] for i in pi],device='cuda')
    losses={}
    for a in ALPHAS:
        h=h2 if a==0 else (h2.float()+a*(h2.float()-h1.float())).to(h2.dtype)
        z=model.lm_head(h[:,idx,:])[0].float()
        nll=-z.log_softmax(-1).gather(-1,target[:,None])[:,0]
        losses[str(a)]={'A':float(nll[::2].mean()),'B':float(nll[1::2].mean())}
    # B cannot choose a different action; it only vetoes A's single choice.
    proposal=min(ALPHAS[1:],key=lambda a:losses[str(a)]['A'])
    va=losses['0.0']['A']-losses[str(proposal)]['A'];vb=losses['0.0']['B']-losses[str(proposal)]['B']
    return dict(alpha=proposal if min(va,vb)>.001 else 0.,proposal=proposal,value_A=va,value_B=vb,value=min(va,vb),losses=losses,token_indices=pi,reason='two_group_input_probe')

@torch.no_grad()
def evaluate(tok,model,pairs,alpha):
    scores=[];effects=[];trace=None
    for ci,(prompt,answer) in enumerate(pairs):
        p,t=encode(tok,prompt,answer);x=torch.tensor([p+t[:-1]],device='cuda')
        intervention=Intervention(alpha);handle=model.model.norm.register_forward_hook(intervention)
        try:_,states,_=model.model(input_ids=x,use_cache=False)
        finally:handle.remove()
        assert intervention.count==4
        positions=list(range(len(p)-1,len(p)+len(t)-1))
        z=[model.lm_head(h[:,positions,:])[0].float() for h in states]
        logits=action_logits(z)['adaptive']
        scores.append(float(logits.log_softmax(-1).gather(-1,torch.tensor(t,device='cuda')[:,None]).sum()))
        effects.append(intervention.effect)
        if ci==0:trace=torch.stack([h[0,len(p)-1,:].float() for h in states]).cpu()
    return scores,effects,trace

def main():
    begin=time.monotonic();deadline=begin+1200
    lock=json.loads((ROOT/'configs/PILOT_LOCK.json').read_text())
    for rel,digest in lock['files'].items():assert sha(ROOT/rel)==digest
    for rel,digest in lock['original_files'].items():assert sha(OLD/rel)==digest
    data=OLD/'data/discovery.jsonl';assert sha(data)==lock['discovery_sha256']
    with data.open(encoding='utf8') as f:rows=[json.loads(l) for l in f][:8]
    assert [r['id'] for r in rows]==lock['ids']
    with (OLD/'results/discovery.jsonl').open(encoding='utf8') as f:
        reference={r['id']:r['views']['official']['sum_scores']['adaptive'] for r in map(json.loads,f)}
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.manual_seed(2021)
    torch.backends.cuda.matmul.allow_tf32=False
    directory=ROOT.parent/'looped_stage1_trajectory_audit_seed2021_v1/models/Ouro-1.4B'
    tok=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    model,info=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.to('cuda').eval();model.requires_grad_(False)
    assert model.model.total_ut_steps==4 and all(not p.requires_grad for p in model.parameters())
    weights_before={n:p._version for n,p in model.named_parameters()}
    meta=json.loads((directory/'snapshot_metadata.json').read_text());assert meta['sha']=='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
    torch.cuda.reset_peak_memory_stats();results=[];parity={}
    # Unique critical-path check: native versus zero feedback, and propagated effects.
    p,t=encode(tok,*rows[0]['variants']['official'][0]);x=torch.tensor([p+t[:-1]],device='cuda')
    _,native,_=model.model(input_ids=x,use_cache=False)
    it=Intervention(0.);handle=model.model.norm.register_forward_hook(it)
    try:_,zero,_=model.model(input_ids=x,use_cache=False)
    finally:handle.remove()
    parity['zero_feedback_exact_all_depths']=all(torch.equal(a,b) for a,b in zip(native,zero))
    assert parity['zero_feedback_exact_all_depths']
    del native,zero,x
    with (ROOT/'results/pilot.jsonl').open('x',encoding='utf8') as out:
        for row in rows:
            assert time.monotonic()<deadline,'Pilot budget exceeded'
            pairs=row['variants']['official'];assert len(set(p[0] for p in pairs))==1
            start=time.monotonic();decision=choose(tok,model,pairs[0][0],row['question']);actions={};traces={}
            for a in ALPHAS:
                assert time.monotonic()<deadline,'Pilot budget exceeded'
                scores,effects,trace=evaluate(tok,model,pairs,a);traces[a]=trace
                actions[str(a)]=dict(sum_scores=scores,intervention_rms=effects,prediction=int(np.argmax(np.array(scores)/row['choice_char_lengths'])))
            drift={str(a):[float((traces[a][r]-traces[0.][r]).square().mean().sqrt()) for r in range(4)] for a in ALPHAS[1:]}
            assert all(v[0]==0 and v[1]>0 and v[2]>0 and v[3]>0 for v in drift.values())
            score_error=max(abs(a-b) for a,b in zip(actions['0.0']['sum_scores'],reference[row['id']]))
            assert score_error<1e-5,score_error
            rec=dict(id=row['id'],decision=decision,actions=actions,downstream_state_rms=drift,baseline_score_error=score_error,selected_prediction=actions[str(decision['alpha'])]['prediction'],gold=row['gold'],elapsed_s=time.monotonic()-start)
            # Gold is attached only after every action and all predictions are finalized.
            out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush();results.append(rec)
            print(json.dumps(dict(completed=len(results),alpha=decision['alpha'],elapsed_s=time.monotonic()-begin)),flush=True)
    assert all(p._version==weights_before[n] for n,p in model.named_parameters())
    torch.cuda.synchronize()
    counts={str(a):sum(r['actions'][str(a)]['prediction']==r['gold'] for r in results) for a in ALPHAS}
    report=dict(scope='Eight exposed development questions: implementation pilot, not efficacy evidence',n=len(results),parity=parity,weights_unchanged=True,nonzero_feedback_propagated_all_questions=True,fixed_action_correct=counts,selected_correct=sum(r['selected_prediction']==r['gold'] for r in results),selected_alpha_counts={str(a):sum(r['decision']['alpha']==a for r in results) for a in ALPHAS},elapsed_s=time.monotonic()-begin,peak_vram_bytes=torch.cuda.max_memory_allocated(),model_revision=meta['sha'],early_stopping_implemented=False,final_tasks_loaded=False,pilot_sha256=sha(ROOT/'results/pilot.jsonl'))
    write('PILOT_COMPLETE.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
