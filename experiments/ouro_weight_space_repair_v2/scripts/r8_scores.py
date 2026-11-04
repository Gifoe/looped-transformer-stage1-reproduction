"""R8 inference-only score geometry. Inputs: complete character-normalized LLs.

No labels, model calls, fitting, answer text or token-level reconstruction.
These are known projection operations used on a new per-question score repair;
orthogonality/halfspace protection does not guarantee preservation of answers.
"""
import numpy as np
from functools import lru_cache

EPS=64*np.finfo(np.float64).eps
METHODS=('keep_r6_axis','nonundo_r6_axis','late_orthogonal','late_unprotected',
         'matched_keep','matched_nonundo','matched_late','preserve_pair_margins')


def finite(*values):
    if any(not np.isfinite(v).all() for v in values):raise ValueError('Nonfinite R8 input/intermediate/output')


def center(value):
    value=np.asarray(value,dtype=np.float64);finite(value)
    shifted=value-value[0];return shifted-shifted.mean()


def norm(value):
    value=np.asarray(value,dtype=np.float64);m=float(np.max(np.abs(value)))
    return 0. if m==0 else m*float(np.sqrt(np.dot(value/m,value/m)))


def unit(value,reference_scale=0.):
    n=norm(value)
    if n==0. or n<=EPS*reference_scale:return None
    return value/n


def projection(value,axis):
    return np.zeros_like(value) if axis is None else float(value@axis)*axis


def bounded_beta(value,direction):
    size=norm(direction)
    if size==0.:return 0.
    n=float(value@(direction/size))
    return 0. if n<=0. else 1. if n>=size else n/size


@lru_cache(maxsize=16)
def partitions(n):
    if n==0:return ((),)
    answer=[]
    for previous in partitions(n-1):
        answer.append(previous+((n-1,),))
        for index in range(len(previous)):
            answer.append(previous[:index]+(previous[index]+(n-1,),)+previous[index+1:])
    return tuple(answer)


def pair_margin_score(a,r,m2,fixed):
    n=len(r);assert 2<=n<=8
    edges=[]
    for i in range(n):
        for j in range(n):
            if r[i]>r[j] and (fixed[i]<fixed[j] or a[i]<a[j]):edges.append((i,j))
    d=center(m2-r);feasible=lambda x:all(x[i]>=x[j] for i,j in edges)
    if feasible(d):best=d.copy();distance=0.
    else:
        best=None;distance=np.inf
        for partition in partitions(n):
            proposal=np.empty_like(d)
            for block in partition:proposal[list(block)]=d[list(block)].mean()
            if not feasible(proposal):continue
            error=norm(proposal-d)
            if error<distance:best=proposal;distance=error
        if best is None:raise RuntimeError('Feasible order projection not found')
    assert feasible(best) and norm(best)<=norm(d)+1e-12*max(norm(d),1.)
    # Edges are actual changed relations, not assumed correct relations.
    top=int(r.argmax());reachable={top}
    for _ in range(n):reachable|={j for i,j in edges if i in reachable}
    output=r+best
    guarantee_error=max((max(0.,(r[i]-r[j])-(output[i]-output[j])) for i,j in edges),default=0.)
    finite(output,best,distance)
    return output,dict(protected_pair_count=len(edges),edges=[list(e) for e in edges],
        raw_increment_norm=norm(d),applied_increment_norm=norm(best),removed_increment_norm=distance,
        conflict_energy_fraction=0. if norm(d)==0 else (distance/norm(d))**2,
        original_winner_locked=len(reachable)==n,raw_increment_feasible=feasible(d),
        max_FP64_margin_rounding_loss=guarantee_error)


def selected_score(a,r,m2,late,prior_late,method,fixed=None):
    if method not in METHODS:raise ValueError('Unknown R8 recipe')
    a,r,m2,late=[np.asarray(v,dtype=np.float64) for v in (a,r,m2,late)]
    assert a.ndim==1 and a.size>=2 and all(v.shape==a.shape for v in [r,m2,late]);finite(a,r,m2,late)
    if method=='preserve_pair_margins':
        fixed=a-late if fixed is None else np.asarray(fixed,dtype=np.float64)
        assert fixed.shape==a.shape;finite(fixed)
        return pair_margin_score(a,r,m2,fixed)
    c=center(r-a);caxis=unit(c,max(norm(center(r)),norm(center(a))))
    d=center(m2-r);pc=projection(d,caxis)
    keep=d-pc
    nonundo=d-projection(d,caxis) if caxis is not None and float(d@caxis)<0. else d.copy()
    diag=dict(protected_axis_active=caxis is not None,protected_axis_norm=norm(c),
        raw_increment_norm=norm(d),raw_increment_dot_axis=float(d@caxis) if caxis is not None else 0.)
    if method in ['keep_r6_axis','nonundo_r6_axis','matched_keep','matched_nonundo']:
        repaired=keep if method in ['keep_r6_axis','matched_keep'] else nonundo
        ratio=1. if norm(d)==0 else norm(repaired)/norm(d)
        assert ratio<=1.+1e-12
        delta=ratio*d if method.startswith('matched_') else repaired
        diag.update(norm_ratio=ratio,repaired_increment_norm=norm(repaired))
    else:
        if prior_late is None:raise ValueError('Late prior P5 is required; no silent fallback')
        h=center(prior_late);assert h.shape==a.shape;finite(h)
        ell=center(late);ell_perp=ell-projection(ell,caxis)
        residual=h-projection(h,caxis)
        if norm(residual)<=np.sqrt(EPS)*norm(h):residual=np.zeros_like(h)
        if norm(ell_perp)<=EPS*norm(ell):ell_perp=np.zeros_like(ell_perp)
        if method=='late_unprotected':
            beta=bounded_beta(ell,h);delta=-beta*h
        else:
            beta=bounded_beta(ell_perp,residual)
            after=ell_perp-beta*residual
            rho=1. if norm(ell_perp)==0 else norm(after)/norm(ell_perp)
            assert rho<=1.+1e-12
            delta=(rho-1.)*ell_perp if method=='matched_late' else -beta*residual
            diag.update(norm_ratio=rho,late_residual_before_norm=norm(ell_perp),late_residual_after_norm=norm(after))
        diag.update(beta=beta,prior_late_norm=norm(h),orthogonal_prior_late_norm=norm(residual))
    output=r+delta;finite(output,delta)
    diag.update(applied_increment_norm=norm(center(delta)),applied_increment_dot_axis=float(center(delta)@caxis) if caxis is not None else 0.)
    return output,diag
