"""Native-distribution KL calibration in one prior direction, target-free.

f(a)=KL(pQ4||softmax(F-a*ebar))-KL(pQ4||softmax(F))+.5*a*a*U.
Standard convex exponential-family calibration, not a correctness objective.
Exactly32 bisections give an alpha interval of width<=2^-32 for interior roots;
this is a bounded numerical solution, not an exact-arithmetic root claim.
"""
import numpy as np

STEPS=32
EPS_MULTIPLIER=64.


def _logp(x):
    x=np.asarray(x,dtype=np.float64);z=x-x.max(axis=-1,keepdims=True)
    return z-np.log(np.exp(z).sum(axis=-1,keepdims=True))


def _center(x,p):
    y=x-x[..., :1];return y-np.sum(p*y,axis=-1,keepdims=True)


def numpy_logits(q1,q4,refs1,refs4,force_zero=False):
    q1,q4,r1,r4=[np.asarray(x) for x in [q1,q4,refs1,refs4]]
    assert q1.dtype==q4.dtype==r1.dtype==r4.dtype==np.float32 and q1.shape==q4.shape and q1.ndim==2
    assert r1.shape==r4.shape and r1.ndim==3 and r1.shape[1:]==q1.shape and r1.shape[0]>0
    assert all(np.isfinite(x).all() for x in [q1,q4,r1,r4])
    g=np.float32(.5)*(q4-q1);fixed=q4+g;p=np.exp(_logp(q4));gc=_center(g.astype('float64'),p)
    ec=_center((np.float32(.5)*(r4-r1)).astype('float64'),p[None])
    if np.array_equal(ec,np.broadcast_to(ec[:1],ec.shape)):ec=ec[:1]
    e=ec.mean(axis=0);U=np.mean(np.sum(p[None]*(ec-e)**2,axis=-1),axis=0);mu=np.sum(p*e,axis=-1);D=np.sum(p*e*e,axis=-1)
    threshold=EPS_MULTIPLIER*np.finfo(float).eps*np.mean(np.max(np.abs(ec),axis=-1)**2,axis=0)
    flat=D<=threshold
    derivative=lambda a:mu-np.sum(np.exp(_logp(fixed.astype('float64')-a[:,None]*e))*e,axis=-1)+a*U
    d0=derivative(np.zeros(len(q1)));d1=derivative(np.ones(len(q1)));low=(d0>=0)|flat;high=(d1<=0)&~low;inside=~(low|high)
    left=np.zeros(len(q1));right=np.ones(len(q1))
    for _ in range(STEPS):
        mid=.5*(left+right);d=derivative(mid);left=np.where(inside&(d<0),mid,left);right=np.where(inside&(d>=0),mid,right)
    a=np.where(low,0.,np.where(high,1.,.5*(left+right)))
    if force_zero:a=np.zeros_like(a)
    exact=fixed.astype('float64')-a[:,None]*e;out=exact.astype('float32');out=np.where((a==0)[:,None],fixed,out)
    lpF=_logp(fixed);objective=lambda z,b:np.sum(p*(lpF-_logp(z)),axis=-1)+.5*b*b*U
    g2=np.sum(p*gc*gc,axis=-1);r2=np.sum(p*(gc-a[:,None]*e)**2,axis=-1);rho=np.sqrt(np.divide(r2,g2,out=np.ones_like(g2),where=g2>0))
    assert not np.any((g2==0)&(r2>threshold))
    matched=(q4.astype('float64')+rho[:,None]*g.astype('float64')).astype('float32');matched=np.where((a==0)[:,None],fixed,matched)
    return out,dict(alpha=a,dispersion=U,target_moment=mu,derivative0=d0,derivative1=d1,root_derivative=derivative(a),
        interior=inside,indistinguishable=flat,alpha_interval_width=np.where(inside,right-left,0.),objective_ideal=objective(exact,a),objective_FP32=objective(out,a),
        max_logit_rounding=np.max(np.abs(out.astype(float)-exact),axis=-1),rho=rho,matched_logits=matched,
        guidance_norm2=g2,residual_norm2=r2,correction_norm2=a*a*D)


def torch_logits(q1,q4,refs1,refs4,permutation,force_zero=False):
    import torch
    assert q1.ndim==2 and q1.shape==q4.shape and refs1.ndim==3 and refs1.shape==refs4.shape and refs1.shape[1:]==q1.shape
    assert all(x.dtype==torch.float32 and x.device==q4.device for x in [q1,q4,refs1,refs4])
    with torch.no_grad():
        assert all(bool(torch.isfinite(x).all()) for x in [q1,q4,refs1,refs4])
        def center(x):
            y=x-x[..., :1];return y-(p*y).sum(-1,keepdim=True)
        g=.5*(q4-q1);fixed=q4+g;p=q4.double().softmax(-1);gc=center(g.double());ec=center((.5*(refs4-refs1)).double())
        if torch.equal(ec,ec[:1].expand_as(ec)):ec=ec[:1]
        e=ec.mean(0);U=(p*(ec-e).square()).sum(-1).mean(0);mu=(p*e).sum(-1);D=(p*e*e).sum(-1)
        threshold=EPS_MULTIPLIER*torch.finfo(torch.float64).eps*ec.abs().amax(-1).square().mean(0);flat=D<=threshold
        def derivative(a):return mu-((fixed.double()-a[:,None]*e).softmax(-1)*e).sum(-1)+a*U
        zero=torch.zeros(len(q1),dtype=torch.float64,device=q1.device);one=torch.ones_like(zero);d0=derivative(zero);d1=derivative(one)
        low=(d0>=0)|flat;high=(d1<=0)&~low;inside=~(low|high);left=zero.clone();right=one.clone()
        # Only interior rows need repeated softmax, but complete vocabulary stays.
        indexes=torch.nonzero(inside).flatten();ef=e[indexes];ff=fixed[indexes].double();uf=U[indexes];mf=mu[indexes]
        lo=left[indexes];hi=right[indexes]
        for _ in range(STEPS):
            mid=.5*(lo+hi);der=mf-((ff-mid[:,None]*ef).softmax(-1)*ef).sum(-1)+mid*uf
            lo=torch.where(der<0,mid,lo);hi=torch.where(der>=0,mid,hi)
        left[indexes]=lo;right[indexes]=hi;a=torch.where(low,zero,torch.where(high,one,.5*(left+right)))
        if force_zero:a=zero.clone()
        exact=fixed.double()-a[:,None]*e;out=exact.float();out=torch.where((a==0)[:,None],fixed,out)
        lpF=fixed.double().log_softmax(-1)
        def objective(z,b):return (p*(lpF-z.double().log_softmax(-1))).sum(-1)+.5*b*b*U
        g2=(p*gc.square()).sum(-1);r2=(p*(gc-a[:,None]*e).square()).sum(-1);assert not bool(((g2==0)&(r2>threshold)).any())
        rho=torch.sqrt(torch.where(g2>0,r2/torch.where(g2>0,g2,torch.ones_like(g2)),torch.ones_like(g2)))
        matched=(q4.double()+rho[:,None]*g.double()).float();matched=torch.where((a==0)[:,None],fixed,matched)
        correction=a[:,None]*e;c2=(p*correction.square()).sum(-1);perm=torch.as_tensor(permutation,device=q1.device)
        assert perm.shape==(q4.shape[-1],) and torch.equal(perm.sort().values,torch.arange(q4.shape[-1],device=q1.device))
        shuffled=center(correction[:,perm]);s2=(p*shuffled.square()).sum(-1);assert not bool(((c2>0)&(s2==0)).any())
        scale=torch.sqrt(torch.where(s2>0,c2/torch.where(s2>0,s2,torch.ones_like(s2)),torch.zeros_like(s2)));shuffled=shuffled*scale[:,None]
        control=(fixed.double()-shuffled).float();control=torch.where((a==0)[:,None],fixed,control)
        rounding=(out.double()-exact).abs().amax(-1);eps=torch.finfo(torch.float64).eps
        numeric_bound=128*eps*q4.shape[-1]*(1+(fixed.double()-fixed.double()[:,:1]).abs().amax(-1)+e.abs().amax(-1))
        # Independent frozen R7 coefficient, not the actual-Fixed Hessian.
        num=(p*gc*e).sum(-1);den=(p*ec*ec).sum(-1).mean(0);old=(num/torch.where(den<=threshold,torch.ones_like(den),den)).clamp(0.,1.);old=torch.where(den<=threshold,zero,old)
        oldlogits=fixed.double()-old[:,None]*e
        fideal=objective(exact,a);factual=objective(out,a);fold=objective(oldlogits,old)
        if not force_zero:assert bool((fideal<=numeric_bound).all()) and bool((fideal<=fold+numeric_bound).all())
        assert bool((factual<=fideal+2*rounding+numeric_bound).all())
        controls=dict(exact_native_KL=out,matched_guidance=matched,shuffled_correction=control)
        diagnostic=dict(alpha=a,old_Fisher_alpha=old,dispersion=U,target_moment=mu,derivative0=d0,derivative1=d1,root_derivative=derivative(a),
            interior=inside,indistinguishable=flat,alpha_interval_width=torch.where(inside,right-left,zero),objective_ideal=fideal,objective_FP32=factual,old_Fisher_objective_ideal=fold,old_Fisher_objective_FP32=objective(oldlogits.float(),old),
            max_logit_rounding=rounding,numerical_bound=numeric_bound,rho=rho,guidance_norm2=g2,residual_norm2=r2,correction_norm2=c2,
            matched_guidance_norm2=rho.square()*g2,shuffled_correction_norm2=(p*shuffled.square()).sum(-1))
        assert all(bool(torch.isfinite(x).all()) for x in list(controls.values())+[v for v in diagnostic.values() if v.dtype!=torch.bool])
        return controls,diagnostic
