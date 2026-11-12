"""Native recurrent state capture primitives. Requires runtime parity gates before full cohort."""
from contextlib import contextmanager
import torch

@contextmanager
def capture(model,positions,depth,identity=False):
    """Capture genuine layer outputs and loop norm states at selected positions."""
    core=model.model
    if not 1<=depth<=4 or len(core.layers)!=24: raise ValueError('invalid architecture/depth')
    if len(set(positions))!=len(positions) or not positions: raise ValueError('positions must be unique and nonempty')
    expected=[(loop,layer) for loop in range(depth) for layer in range(24)]
    events=[]; layer_states={}; norm_states=[]; handles=[]; active={}
    def pre(i):
        def hook(module,args,kwargs):
            loop=kwargs.get('current_ut')
            if not isinstance(loop,int): raise RuntimeError('missing native current_ut')
            event=(loop,i)
            if len(events)>=len(expected) or event!=expected[len(events)]: raise RuntimeError(f'unexpected layer sequence: {event}')
            if i==0 and loop>0:
                actual=args[0][0,positions,:].detach().cpu()
                if not torch.equal(actual,norm_states[-1]): raise RuntimeError('loop norm to next layer input mismatch')
            events.append(event);active[i]=loop
        return hook
    def post(i):
        def hook(module,args,kwargs,out):
            if not torch.is_tensor(out) or out.ndim!=3: raise RuntimeError('unexpected layer output')
            loop=active.pop(i);layer_states[f'loop{loop+1}_layer{i:02d}']=out[0,positions,:].detach().cpu().clone()
            if identity:
                replacement=out.clone()
                replacement[0,positions,:]=out[0,positions,:].clone()
                return replacement
        return hook
    def norm_hook(module,args,out):
        if not torch.is_tensor(out) or out.ndim!=3 or len(events)!=24*(len(norm_states)+1): raise RuntimeError('norm placement mismatch')
        before=args[0][0,positions,:].detach().cpu()
        key=f'loop{len(norm_states)+1}_layer23'
        if not torch.equal(before,layer_states[key]): raise RuntimeError('last layer to norm input mismatch')
        norm_states.append(out[0,positions,:].detach().cpu().clone())
    old=core.total_ut_steps
    try:
        core.total_ut_steps=depth
        for i,layer in enumerate(core.layers):
            if layer.self_attn.layer_idx!=i: raise RuntimeError('layer index mismatch')
            handles.append(layer.register_forward_pre_hook(pre(i),with_kwargs=True))
            handles.append(layer.register_forward_hook(post(i),with_kwargs=True))
        handles.append(core.norm.register_forward_hook(norm_hook))
        yield {'layer_states':layer_states,'loop_post_norm':norm_states,'events':events}
        if events!=expected or len(norm_states)!=depth or active: raise RuntimeError('incomplete capture')
    finally:
        for h in handles: h.remove()
        core.total_ut_steps=old

def candidate_logprob(model,norm_state,target_ids,predecessor_indices):
    """Use original full-vocabulary BF16 head, then FP32 full-vocabulary log-softmax."""
    if len(target_ids)!=len(predecessor_indices): raise ValueError('target/predecessor mismatch')
    h=norm_state[predecessor_indices].unsqueeze(0).to(model.lm_head.weight.device)
    if h.dtype!=torch.bfloat16: raise TypeError('expected native BF16 states')
    logits=model.lm_head(h)[0].float()
    target=torch.tensor(target_ids,device=logits.device,dtype=torch.long)
    values=logits.log_softmax(-1).gather(-1,target[:,None]).squeeze(-1)
    if not bool(torch.isfinite(values).all()): raise ValueError('nonfinite log probabilities')
    return float(values.sum())
