"""Frozen-weight inverse-question evidence, label-free score fusion.

The auxiliary likelihood is not a Bayes decomposition of the primary prompt.
Equal norm is a bounded hypothesis, not a calibrated reliability estimate.
"""
import numpy as np
from r8_scores import center, norm, finite, pair_margin_score, EPS

ELIGIBLE=('reverse4_fusion','reverse4_protected','reverseFixed_protected')
CONTROL='reverse4_shuffled_control'


def evidence(r,b):
    r,b=np.asarray(r,dtype=np.float64),np.asarray(b,dtype=np.float64)
    assert r.shape==b.shape and r.ndim==1 and len(r)>=2
    finite(r,b)
    d=center(b); size=norm(d); scale=max(float(np.max(np.abs(b))),1.)
    unresolved=size<=EPS*scale
    u=np.zeros_like(d) if unresolved else d*(norm(center(r))/size)
    finite(u)
    return u,dict(auxiliary_centered_norm=size,primary_centered_norm=norm(center(r)),
                  auxiliary_unresolved=bool(unresolved),equal_norm_increment=norm(u))


def score(a,r,fixed,aux4,auxFixed):
    a,r,fixed=[np.asarray(x,dtype=np.float64) for x in [a,r,fixed]]
    assert a.shape==r.shape==fixed.shape;finite(a,r,fixed)
    u,d=evidence(r,aux4);uf,df=evidence(r,auxFixed)
    protected,pd=pair_margin_score(a,r,r+u,fixed)
    pf,pfd=pair_margin_score(a,r,r+uf,fixed)
    permutation=np.random.default_rng(2021).permutation(len(r))
    control,cd=pair_margin_score(a,r,r+u[permutation],fixed)
    target=norm(center(protected-r));cn=norm(center(control-r))
    resolved=cn>EPS*max(norm(u),1.)
    matched=resolved or target==0.
    # Scaling by a nonnegative scalar preserves the same order cone. If zero
    # makes matching impossible, keep zero and explicitly mark that question.
    control=r+(np.zeros_like(r) if not resolved else (target/cn)*center(control-r))
    matcherr=abs(norm(center(control-r))-target)
    values=dict(reverse4_fusion=r+u,reverse4_protected=protected,
                reverseFixed_protected=pf,reverse4_shuffled_control=control)
    finite(*values.values())
    return values,dict(reverse4=d,reverseFixed=df,protected=pd,protectedFixed=pfd,
        control_projection=cd,control_permutation=permutation.tolist(),
        control_norm_match_possible=bool(matched),control_target_norm=target,
        control_applied_norm=norm(center(control-r)),control_norm_absolute_error=matcherr)
