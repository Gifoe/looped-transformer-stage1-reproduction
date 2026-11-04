"""R20 unchanged native Fisher projection plus raw-demo and shuffle controls."""
import numpy as np
from r7_math import token_logits_numpy

def numpy_controls(q1,q4,b1,b4,d1,d4,permutation):
    primary,diag=token_logits_numpy(q1,q4,b1[None],b4[None])
    demo,dd=token_logits_numpy(q1,q4,d1[None],d4[None])
    shuffled,sd=token_logits_numpy(q1,q4,b1[None],b4[None],mode='shuffle',permutation=permutation)
    z=q4.astype(np.float64);p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    def center(x):
        x=x-x[...,:1];return x-(p*x).sum(-1,keepdims=True)
    ec=center((np.float32(.5)*(b4-b1)).astype(np.float64));d=center((np.float32(.5)*(d4-d1)).astype(np.float64))
    c=diag['alpha'][...,None]*ec;norm=(p*c*c).sum(-1);dn=(p*d*d).sum(-1);threshold=64*np.finfo(np.float64).eps*np.max(np.abs(d),axis=-1)**2
    unmatched=(norm>0)&(dn<=threshold)
    scale=np.zeros_like(dn);np.divide(norm,dn,out=scale,where=(dn>threshold));matched=d*np.sqrt(scale)[...,None]
    fixed=q4+np.float32(.5)*(q4-q1);control=(fixed.astype(np.float64)-matched).astype(np.float32);control=np.where((norm==0)[...,None],fixed,control)
    assert all(np.isfinite(x).all() for x in [matched,control])
    return dict(primary=primary,demo=demo,raw_demo_matched=control,shuffle=shuffled),dict(alpha=diag['alpha'],demo_alpha=dd['alpha'],unmatched=unmatched,main_norm2=norm,matched_norm2=(p*matched*matched).sum(-1),shuffle_norm2=sd['applied_direction_norm2'])

def torch_controls(q1,q4,b1,b4,d1,d4,permutation):
    import torch
    from r7_math import token_logits
    primary,diag=token_logits(q1,q4,b1.unsqueeze(0),b4.unsqueeze(0))
    demo,dd=token_logits(q1,q4,d1.unsqueeze(0),d4.unsqueeze(0))
    shuffled,sd=token_logits(q1,q4,b1.unsqueeze(0),b4.unsqueeze(0),mode='shuffle',permutation=permutation)
    p=torch.softmax(q4.double(),dim=-1)
    def center(x):
        x=x-x[...,:1];return x-(p*x).sum(-1,keepdim=True)
    ec=center((.5*(b4-b1)).double());d=center((.5*(d4-d1)).double());c=diag['alpha'].unsqueeze(-1)*ec
    norm=(p*c*c).sum(-1);dn=(p*d*d).sum(-1);threshold=64*torch.finfo(torch.float64).eps*d.abs().amax(-1).square();unmatched=(norm>0)&(dn<=threshold)
    scale=torch.where(dn>threshold,norm/torch.where(dn>threshold,dn,torch.ones_like(dn)),torch.zeros_like(dn));matched=d*scale.sqrt().unsqueeze(-1)
    fixed=q4+.5*(q4-q1);control=(fixed.double()-matched).float();control=torch.where((norm==0).unsqueeze(-1),fixed,control)
    assert bool(torch.isfinite(control).all())
    return dict(primary=primary,demo=demo,raw_demo_matched=control,shuffle=shuffled),dict(alpha=diag['alpha'],demo_alpha=dd['alpha'],unmatched=unmatched,main_norm2=norm,matched_norm2=(p*matched*matched).sum(-1),shuffle_norm2=sd['applied_direction_norm2'])

def synthetic_checks():
    rng=np.random.default_rng(2021);a=[rng.normal(size=(3,19)).astype(np.float32) for _ in range(6)];perm=rng.permutation(19)
    out,diag=numpy_controls(*a,perm)
    assert not diag['unmatched'].any()
    for k in ['matched_norm2','shuffle_norm2']:assert np.max(np.abs(diag[k]-diag['main_norm2']))<1e-10
    zero=a.copy();zero[3]=zero[2].copy();zo,zd=numpy_controls(*zero,perm);fixed=a[1]+np.float32(.5)*(a[1]-a[0]);assert all(np.array_equal(v,fixed) for k,v in zo.items() if k!='demo')
    extra=[np.concatenate([x,rng.normal(size=(1,19)).astype(np.float32)]) for x in a];ext,_=numpy_controls(*extra,perm)
    assert all(np.array_equal(out[k],ext[k][:3]) for k in out)
    bad=a.copy();bad[0]=bad[0].copy();bad[0][0,0]=np.nan
    try:numpy_controls(*bad,perm)
    except ValueError:pass
    else:raise AssertionError('Nonfinite accepted')
    # Exact representable common shifts test coefficient gauge invariance.
    gauge=[np.round(x*8)/8 for x in a];_,gd=numpy_controls(*gauge,perm);_,gg=numpy_controls(*(x+np.float32(2) for x in gauge),perm)
    assert np.max(np.abs(gd['alpha']-gg['alpha']))<1e-12
    return dict(status='passed_numpy_CPU',zero_reference=True,nonfinite_rejected=True,gauge=True,norm_match=True,candidate_extension=True,Torch_parity='pending_CPU_server')
if __name__=='__main__':
    import json
    print(json.dumps(synthetic_checks()))
