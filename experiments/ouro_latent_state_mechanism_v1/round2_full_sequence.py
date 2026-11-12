"""ROUND 2 draft: full-sequence intervention primitives, NOT an accepted gate.

Callers must separately validate exact baseline scores, anchors, resources,
provenance and no-cache behavior before using these primitives scientifically.
"""
from contextlib import contextmanager
import torch


def _validate_states(k1,k2,readout):
    if not (torch.is_tensor(k1) and torch.is_tensor(k2) and k1.ndim==3 and k1.shape==k2.shape and k1.shape[0]==1):
        raise ValueError('states must be matching batch-one full-sequence tensors')
    if k1.dtype!=k2.dtype or k1.device!=k2.device:raise ValueError('state dtype/device mismatch')
    if not (torch.isfinite(k1).all() and torch.isfinite(k2).all()):raise ValueError('nonfinite state')
    positions=list(readout)
    if not positions or len(set(positions))!=len(positions) or any(not isinstance(i,int) or i<0 or i>=k1.shape[1] for i in positions):
        raise ValueError('invalid readout positions')
    return positions


def compose_state(k1,k2,readout,condition):
    """All non-readout positions form C; return cloned K2 with requested K1 blocks."""
    positions=_validate_states(k1,k2,readout)
    if condition not in ('K2C_K2R','K1C_K2R','K2C_K1R','K1C_K1R'):raise ValueError('unknown condition')
    out=k2.clone();mask=torch.ones(k1.shape[1],dtype=torch.bool,device=k1.device);mask[positions]=False
    if condition in ('K1C_K2R','K1C_K1R'):out[:,mask,:]=k1[:,mask,:]
    if condition in ('K2C_K1R','K1C_K1R'):out[:,positions,:]=k1[:,positions,:]
    return out


def rescale_direction(source,reference):
    """Direction of source, target norm of reference; return BF16 state + measured norms."""
    if source.shape!=reference.shape or source.dtype!=reference.dtype or source.device!=reference.device or source.ndim!=3:
        raise ValueError('norm control shape/dtype/device mismatch')
    if source.dtype!=torch.bfloat16:raise TypeError('norm controls require BF16')
    a=source.float();b=reference.float();na=torch.linalg.vector_norm(a,dim=-1,keepdim=True);nb=torch.linalg.vector_norm(b,dim=-1,keepdim=True)
    if not (torch.isfinite(na).all() and torch.isfinite(nb).all()) or (na==0).any():raise ValueError('nonfinite or zero direction norm')
    mixed=(a*(nb/na)).to(source.dtype)
    actual=torch.linalg.vector_norm(mixed.float(),dim=-1)
    return mixed,{'source_norm':na.squeeze(-1).detach().cpu(),'reference_norm':nb.squeeze(-1).detach().cpu(),'actual_bf16_norm':actual.detach().cpu()}


@contextmanager
def full_boundary(model,depth,*,condition=None,k1=None,k2_reference=None,readout=(),layer_index=11):
    """Capture full layer-output states; optionally intervene in second pass only.

    Assumes full sequence is batch one, no mask/position modification, and all
    targets are scored in native no-cache forward. Preconditions fail closed.
    """
    core=model.model
    if depth not in (1,2) or len(core.layers)!=24 or layer_index not in (11,23):raise ValueError('invalid depth or layer')
    if condition is not None and (depth!=2 or k1 is None or k2_reference is None):raise ValueError('patch requires K1 and K2 full states at depth 2')
    layer=core.layers[layer_index]
    if layer.self_attn.layer_idx!=layer_index:raise ValueError('layer identity mismatch')
    old=core.total_ut_steps;handles=[];events=[];states={};active=[]
    def model_pre(module,args,kwargs):
        if kwargs.get('labels') is not None or kwargs.get('use_cache') is not False or kwargs.get('past_key_values') is not None:
            raise RuntimeError('must use labels=None, use_cache=False, no past key values')
    def pre(module,args,kwargs):
        loop=kwargs.get('current_ut')
        if not isinstance(loop,int) or loop!=len(events) or loop>=depth or active:raise RuntimeError('bad current_ut sequence')
        events.append(loop);active.append(loop)
    def post(module,args,kwargs,out):
        if len(active)!=1 or not torch.is_tensor(out) or out.ndim!=3:raise RuntimeError('bad boundary output')
        loop=active.pop();states[loop]=out.detach().clone()
        if condition is not None and loop==1:
            if not torch.equal(out,k2_reference):raise RuntimeError('K2 boundary differs from reference')
            return compose_state(k1,out,readout,condition)
    try:
        core.total_ut_steps=depth
        handles.append(model.register_forward_pre_hook(model_pre,with_kwargs=True))
        handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
        handles.append(layer.register_forward_hook(post,with_kwargs=True))
        yield states
        if events!=list(range(depth)) or active or set(states)!=set(range(depth)):raise RuntimeError('incomplete boundary events')
    finally:
        for h in handles:h.remove()
        core.total_ut_steps=old
