"""Full-vocabulary context-contrast diagnostics, never a repair or controller.

I(p)=E_p[log p(Q4)-log p(reference4)] is an expectation of a conditional
log-likelihood ratio. It is not ground-truth utility or mutual information.
No target IDs, labels, candidate text, thresholds or fitted parameters enter.
"""
import numpy as np


def logp_np(x):
    x=np.asarray(x,dtype=np.float64);u=x-x.max(axis=-1,keepdims=True)
    return u-np.log(np.exp(u).sum(axis=-1,keepdims=True))


def context_np(q4,fixed,repaired,reference4):
    q,f,n,r=[np.asarray(x) for x in [q4,fixed,repaired,reference4]]
    assert q.ndim==2 and q.shape==f.shape==n.shape and r.ndim==3 and r.shape[1:]==q.shape
    assert r.shape[0]>0 and all(np.isfinite(x).all() for x in [q,f,n,r])
    lq,lf,ln,lr=[logp_np(x) for x in [q,f,n,r]];pf=np.exp(lf);pn=np.exp(ln);s=lq[None]-lr
    static=np.sum(pf[None]*s,axis=-1);change=np.sum((pn-pf)[None]*s,axis=-1)
    kl=np.sum(pn*(ln-lf),axis=-1)
    assert np.isfinite(static).all() and np.isfinite(change).all() and np.all(kl>=-1e-12)
    return dict(static=static,change=change,repair_KL=kl)


def context_torch(q4,fixed,repaired,reference4):
    import torch
    assert q4.ndim==2 and q4.shape==fixed.shape==repaired.shape and reference4.ndim==3 and reference4.shape[1:]==q4.shape
    assert reference4.shape[0]>0
    with torch.no_grad():
        q,f,n,r=[x.detach().double() for x in [q4,fixed,repaired,reference4]]
        assert all(bool(torch.isfinite(x).all()) for x in [q,f,n,r])
        lq,lf,ln,lr=[x.log_softmax(-1) for x in [q,f,n,r]];pf=lf.exp();pn=ln.exp();s=lq.unsqueeze(0)-lr
        static=(pf.unsqueeze(0)*s).sum(-1);change=((pn-pf).unsqueeze(0)*s).sum(-1);kl=(pn*(ln-lf)).sum(-1)
        assert bool(torch.isfinite(static).all()) and bool(torch.isfinite(change).all()) and bool((kl>=-1e-12).all())
        return dict(static=static,change=change,repair_KL=kl)


def synthetic_checks():
    rng=np.random.default_rng(2021);q=rng.normal(size=(5,13)).astype('float32');f=q+rng.normal(size=q.shape).astype('float32')*.2;n=f+rng.normal(size=q.shape).astype('float32')*.1;r=rng.normal(size=(2,)+q.shape).astype('float32')
    base=context_np(q,f,n,r);zero=context_np(q,f,f,r);assert np.array_equal(zero['change'],np.zeros((2,5)))
    common=context_np(q,f,n,np.stack([q,q]));assert np.max(np.abs(common['static']))==0 and np.max(np.abs(common['change']))==0
    # Add constants in FP64, so this tests invariance without FP32 re-rounding.
    shifted=context_np(q.astype('float64')+2,f.astype('float64')-3,n.astype('float64')+5,r.astype('float64')+7)
    assert max(np.max(np.abs(base[k]-shifted[k])) for k in base)<1e-12
    perm=rng.permutation(q.shape[-1]);other=context_np(q[:,perm],f[:,perm],n[:,perm],r[:,:,perm]);assert max(np.max(np.abs(base[k]-other[k])) for k in base)<1e-12
    dup=context_np(q,f,n,r[:1].repeat(3,axis=0));assert np.array_equal(dup['change'][0],dup['change'][1])
    return dict(zero_repair=True,identical_context=True,gauge_invariance=True,vocabulary_permutation=True,duplicate_reference=True)


def witness():
    from r7_math import token_logits_numpy
    q4=np.array([[0.,2.,-1.,1.]],dtype='float32');e=np.array([[-.5,.5,.5,0.]],dtype='float32');q1=q4-2*e;fixed=q4+e
    # Equal prior-loop increments at non-target coordinates1 and2 allow their
    # absolute reference logits to be swapped without changing any e statistic.
    changed=np.exp(logp_np(q4))[0]-np.exp(logp_np(fixed))[0]
    middle=q4[0].astype('float64').copy();middle[1:3]=.5
    middle[0]+=float(changed@(q4[0]-middle))/changed[0]
    left=middle.copy();left[1]=2.5;left[2]=-1.5;right=left.copy();right[[1,2]]=right[[2,1]]
    outputs=[];statistics=[];prior_lp=[];values=[]
    for row in [left,right]:
        r4=row[None].astype('float32');r1=r4-2*e
        out,d=token_logits_numpy(q1,q4,r1[None],r4[None]);outputs.append(out);statistics.append(d)
        prior_lp.append([float(logp_np(z)[0,0]) for z in [r1,r4,r4+.5*(r4-r1)]])
        values.append(float(context_np(q4,fixed,out,r4[None])['change'][0,0]))
    assert np.array_equal(outputs[0],outputs[1])
    maximum=0.
    for k in statistics[0]:
        x,y=statistics[0][k],statistics[1][k]
        if isinstance(x,np.ndarray):assert np.array_equal(x,y);maximum=max(maximum,float(np.max(np.abs(x.astype(float)-y.astype(float)))))
    assert np.max(np.abs(np.asarray(prior_lp[0])-prior_lp[1]))<1e-12 and values[0]*values[1]<0
    return dict(status='nonidentifiability_witness',vocabulary=4,target_coordinate=0,permuted_non_target_coordinates=[1,2],
        actual_R7_numpy_projected_logits_bit_equal=True,all_projection_statistic_arrays_bit_equal=True,max_projection_statistic_error=maximum,
        canonical_reference_target_logprob_max_error=float(np.max(np.abs(np.asarray(prior_lp[0])-prior_lp[1]))),context_change_two_worlds=values,
        claim='Absolute non-target reference logits matter. Stored R7 scalar projection summaries and target likelihoods alone do not identify context-contrast retention.',
        scope='Mathematical/canonical-FP64 likelihood equality; not a claim every possible GPU reduction yields identical serialized bits. Known weights plus re-execution can recover the missing information.')


if __name__=='__main__':
    import json
    print(json.dumps(dict(checks=synthetic_checks(),witness=witness()),indent=2))
