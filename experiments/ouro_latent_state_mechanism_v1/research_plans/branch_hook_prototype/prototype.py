"""CPU prototype only; output hooks capture/patch post-norm residual increments.
Source modeling_ouro.py SHA256 c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413, lines 368-423, 592-604.
Replacing both branches does not replace the independently retained residual stream.
"""
from contextlib import contextmanager
import torch,hashlib

@contextmanager
def branch_hook(layer, physical_idx, expected_ut, branch, positions, donor=None, donor_binding=None):
    if type(expected_ut) is not int or expected_ut<0: raise ValueError('expected_ut')
    if branch not in ('attention','mlp'): raise ValueError('unknown branch')
    if not isinstance(physical_idx,int) or physical_idx<0: raise ValueError('layer index')
    actual=layer.self_attn.layer_idx
    if actual!=physical_idx: raise ValueError('physical layer mismatch')
    target=layer.input_layernorm_2 if branch=='attention' else layer.post_attention_layernorm_2
    state={'active':False,'visits':0,'observed':None}; handles=[]
    def before(module,args,kwargs):
        if state['active']: raise RuntimeError('reentrant layer')
        state['active']=True; state['current_ut']=kwargs.get('current_ut')
        if type(state['current_ut']) is not int or state['current_ut']<0: raise ValueError('current_ut')
    def capture(module,args,out):
        if not state['active']: raise RuntimeError('branch outside physical layer')
        if state['current_ut']!=expected_ut: return None
        if state['visits']: raise RuntimeError('duplicate loop/layer branch')
        if not torch.is_tensor(out) or out.ndim!=3 or out.shape[0]!=1 or not torch.isfinite(out).all(): raise ValueError('unsupported or nonfinite output')
        idx=list(positions)
        if not idx or len(idx)!=len(set(idx)) or any(not isinstance(i,int) or i<0 or i>=out.shape[1] for i in idx): raise ValueError('positions')
        state['visits']+=1;state['observed']=out.detach().clone()
        if donor is None: return None
        if donor.shape!=out.shape or donor.dtype!=out.dtype or donor.device!=out.device or not torch.isfinite(donor).all(): raise ValueError('donor mismatch')
        expected={'physical_idx':physical_idx,'branch':branch,'donor_ut':donor_binding.get('donor_ut') if isinstance(donor_binding,dict) else None,'input_sha256':donor_binding.get('input_sha256') if isinstance(donor_binding,dict) else None,'donor_sha256':hashlib.sha256(donor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
        if donor_binding!=expected or expected['donor_ut'] not in (0,1) or not isinstance(expected['input_sha256'],str) or len(expected['input_sha256'])!=64: raise ValueError('unbound donor')
        result=out.clone(); result[:,idx,:]=donor[:,idx,:]
        return result
    def after(module,args,kwargs,out): state['active']=False
    try:
        handles.append(layer.register_forward_pre_hook(before,with_kwargs=True))
        handles.append(target.register_forward_hook(capture))
        handles.append(layer.register_forward_hook(after,with_kwargs=True,always_call=True))
        yield state
        if state['visits']!=1: raise RuntimeError('target branch must be observed exactly once')
    finally:
        for h in handles: h.remove()
