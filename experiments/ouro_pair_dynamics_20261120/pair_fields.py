"""Causal pair vector fields and operators that expose only current state."""
from pathlib import Path
import sys

import torch
from torch.func import jvp

from norm_flow import NormFlow

ROOT=Path(__file__).resolve().parent
EMBED=ROOT.parent/"ouro_continuous_embedding_20261120"
sys.path.insert(0,str(EMBED))
from dynamics_embedding import EmbeddingOperators, anderson, integrate, norm, picard, rel, timed, weight_hash, write_json
from embedding_run import load_runtime

COEFFICIENTS={"base":(0.,0.),"self":(0.,1.),"order":(1.,0.),"full":(1.,1.),"reverse":(-1.,1.)}
CLOCKS=("linear","slow_fast","fast_slow")


class PairOps(EmbeddingOperators):
    """The public prediction operator captures masks/positions during K1 only."""
    def __init__(self,model,items):
        self.model=model
        self.core=model.model
        self.dtype=next(model.parameters()).dtype
        self.encodings=items
        self.layers=list(self.core.layers)
        assert len(self.layers)==24
        self.lengths=torch.as_tensor([len(w["ids"]) for w in items],device="cuda")
        width=int(self.lengths.max())
        self.ids=torch.tensor([w["ids"]+[0]*(width-len(w["ids"])) for w in items],device="cuda")
        self.rows=torch.arange(len(items),device="cuda")
        self.valid=torch.arange(width,device="cuda")[None]<self.lengths[:,None]
        self.positions=torch.as_tensor([w["read"] for w in items],device="cuda")
        self.candidates=torch.as_tensor([w["content"] for w in items],device="cuda")
        self.counts={k:0 for k in ("attention_calls","mlp_calls","native_loops","loop_field_calls",
                                  "jvp","vjp","inverse_iterations")}
        self.kwargs={}
        hooks=[]
        def capture(i,kw):
            assert kw["past_key_value"] is None and not kw["use_cache"]
            self.kwargs[i]=kw
        for i,layer in enumerate(self.layers):
            hooks.append(layer.register_forward_pre_hook(lambda _,_a,k,i=i:capture(i,k),with_kwargs=True))
        try:
            self.core.total_ut_steps=1
            _,states,_=self.core(input_ids=self.ids,use_cache=False)
            self.start=states[0]
        finally:
            for hook in hooks:
                hook.remove()
        assert len(self.kwargs)==24
        self.norm_flow=NormFlow(self.core.norm)


def derivatives(op,layer,x,order,self_weight):
    a=lambda z:op.branch(2*layer,z).float()
    m=lambda z:op.branch(2*layer+1,z).float()
    u,v=a(x),m(x)
    combined=u+v
    if order==self_weight==0:
        return combined,None
    second=op.deriv(m,x,order*u-self_weight*v)+op.deriv(a,x,-order*v-self_weight*u)
    return combined,second


def bracket(op,layer,x):
    _,y=derivatives(op,layer,x,1.,0.)
    return y


def pair_native(op,layer,x,alpha=1,swap=False):
    indices=(2*layer+1,2*layer) if swap else (2*layer,2*layer+1)
    h=x.to(op.dtype)
    for i in indices:
        h=h+alpha*op.branch(i,h)
    return h


def prefix_inverse(op,f,z,t,steps=6):
    x=z
    residual=[]
    maps=[]
    for _ in range(steps):
        op.counts["inverse_iterations"]+=1
        y=z-t*f(x)
        residual.append((y-x)*op.valid[...,None])
        maps.append(y)
        e=torch.stack(residual,dim=1)
        gram=torch.einsum("bktd,bltd->btkl",e,e).cumsum(1)
        n=gram.shape[-1]
        regularizer=(gram.diagonal(dim1=-2,dim2=-1).mean(-1)*1e-5).clamp_min(1e-12)
        gram=gram+regularizer[...,None,None]*torch.eye(n,device=z.device)
        w=torch.linalg.solve_ex(gram,torch.ones_like(gram[...,:1]))[0].squeeze(-1)
        x=(torch.stack(maps,dim=2)*(w/w.sum(-1,keepdim=True))[...,None]).sum(2)
    return f(x)


def pair(op,layer,x,method,alpha=1.,steps=2,scheme="rk4",swap=False,clock="linear",scale=1.):
    h=x.float()
    if method=="native":
        return pair_native(op,layer,x,alpha,swap)
    i,j=(2*layer+1,2*layer) if swap else (2*layer,2*layer+1)
    if method=="taylor":
        a=lambda z:op.branch(i,z).float()
        m=lambda z:op.branch(j,z).float()
        u=a(h)
        return h+alpha*(u+m(h))+alpha**2*op.deriv(m,h,u)
    if method in ("picard","anderson","prefix_anderson","hybrid","branch_flow","branch_bea"):
        for k in (i,j):
            f=lambda z:op.branch(k,z).float()
            if method=="branch_flow":
                h=integrate(lambda y,t:f(y),h,alpha,steps,scheme)
            elif method=="branch_bea":
                h=integrate(lambda y,t:f(y)-alpha/2*op.deriv(f,y,f(y)),h,alpha,steps,scheme)
            else:
                b=f(h)
                mid=h+alpha*b/2
                if method=="prefix_anderson":
                    h=h+alpha*prefix_inverse(op,f,mid,alpha/2)
                else:
                    fn=picard if method=="picard" else anderson
                    _,direction,residual=fn(op,f,mid,alpha/2)
                    h=h+alpha*(torch.where((residual>2e-5)[:,None,None],b,direction) if method=="hybrid" else direction)
        return h
    c,d=COEFFICIENTS["base" if method=="base_fine" else ("full" if method=="full_scaled" else method)]
    c=-c if swap else c
    steps=steps*2 if method=="base_fine" else steps

    def field(y,t):
        v,term=derivatives(op,layer,y,c,d)
        if term is not None:
            v=v+alpha*scale/2*term
        if clock=="slow_fast":
            v=v*(.5+t/alpha)
        elif clock=="fast_slow":
            v=v*(1.5-t/alpha)
        return v
    return integrate(field,h,alpha,steps,scheme)
