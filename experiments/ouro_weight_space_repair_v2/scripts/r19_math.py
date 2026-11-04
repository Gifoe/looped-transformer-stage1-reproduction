"""Prior protection by the current Q-minus-N0 last trained-loop innovation.

Standard constrained projections, not verified semantic directions. The
reference-only axis is a mandatory ineligible falsification control.
"""
import numpy as np
EPS=64.*np.finfo(np.float64).eps


def numpy_primary(q1,q3,q4,r1,r3,r4):
    q1,q3,q4,r1,r3,r4=[np.asarray(x) for x in [q1,q3,q4,r1,r3,r4]]
    assert all(x.dtype==np.float32 for x in [q1,q3,q4,r1,r3,r4]) and r1.shape==r3.shape==r4.shape and r1.shape[0]>=1
    assert all(np.array_equal(x,np.broadcast_to(x[:1],x.shape)) for x in [r1,r3,r4])
    z=q4.astype(float);p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    def center(x):
        y=x-x[:,:1];return y-np.sum(p*y,axis=-1,keepdims=True)
    g=.5*(q4-q1);F=q4+g;gc=center(g.astype(float));ec=center((.5*(r4[0]-r1[0])).astype(float))
    s=center((q4-q3).astype(float)-(r4[0]-r3[0]).astype(float));s2=np.sum(p*s*s,axis=-1);bad=s2<=EPS*np.max(np.abs(s),axis=-1)**2
    b=np.divide(np.sum(p*ec*s,axis=-1),s2,out=np.zeros_like(s2),where=~bad);residual=ec-b[:,None]*s;D=np.sum(p*residual*residual,axis=-1);threshold=EPS*np.max(np.abs(ec),axis=-1)**2
    a=np.clip(np.divide(np.sum(p*gc*residual,axis=-1),D,out=np.zeros_like(D),where=D>threshold),0,1)
    out=(F.astype(float)-a[:,None]*residual).astype('float32');out=np.where((a==0)[:,None],F,out)
    return out,dict(alpha=a,b=b)


def torch_logits(q1,q3,q4,refs1,refs3,refs4,permutation,force_zero=False):
    import torch
    from r7_math import token_logits
    assert q1.shape==q3.shape==q4.shape and refs1.shape==refs3.shape==refs4.shape and refs1.shape[0]>=1 and refs1.shape[1:]==q1.shape
    assert all(torch.equal(x,x[:1].expand_as(x)) for x in [refs1,refs3,refs4]),'Pilot supports only one reference or identical duplicates'
    assert all(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) for x in [q1,q3,q4,refs1,refs3,refs4])
    with torch.no_grad():
        p=q4.double().softmax(-1)
        def center(x):
            y=x-x[:,:1];return y-(p*y).sum(-1,keepdim=True)
        def dot(x,y):return (p*x*y).sum(-1)
        g=.5*(q4-q1);F=q4+g;gc=center(g.double());ec=center((.5*(refs4[0]-refs1[0])).double())
        u=center((q4-q3).double());v=center((refs4[0]-refs3[0]).double());s=center((q4-q3).double()-(refs4[0]-refs3[0]).double())
        threshold=EPS*ec.abs().amax(-1).square();old,od=token_logits(q1,q4,refs1,refs4)
        def protect(axis):
            axis2=dot(axis,axis);bad=axis2<=EPS*axis.abs().amax(-1).square();b=torch.where(bad,torch.zeros_like(axis2),dot(ec,axis)/torch.where(bad,torch.ones_like(axis2),axis2))
            residual=ec-b[:,None]*axis;D=dot(residual,residual);a=(dot(gc,residual)/torch.where(D<=threshold,torch.ones_like(D),D)).clamp(0,1);a=torch.where(D<=threshold,torch.zeros_like(a),a)
            if force_zero:a=torch.zeros_like(a)
            correction=a[:,None]*residual;z=(F.double()-correction).float();z=torch.where((a==0)[:,None],F,z)
            # Explicit unresolved-axis reduction to frozen original M2.
            if not force_zero:z=torch.where(bad[:,None],old,z);a=torch.where(bad,od['alpha'],a);correction=torch.where(bad[:,None],a[:,None]*ec,correction)
            return z,dict(alpha=a,b=b,axis2=axis2,axis_unresolved=bad,residual=residual,correction=correction)
        main,d=protect(s);ref,rd=protect(v);perm=torch.as_tensor(permutation,device=q4.device)
        assert perm.shape==(q4.shape[-1],) and torch.equal(perm.sort().values,torch.arange(q4.shape[-1],device=q4.device))
        axis_shuffle,ad=protect(center(s[:,perm]))
        correction=d['correction'];cp=center(correction);g2=dot(gc,gc);r2=dot(gc-cp,gc-cp);c2=dot(cp,cp)
        rho=torch.sqrt(torch.where(g2>0,r2/torch.where(g2>0,g2,torch.ones_like(g2)),torch.ones_like(g2)))
        matched=(q4.double()+rho[:,None]*g.double()).float();matched=torch.where((d['alpha']==0)[:,None],F,matched)
        shuffled=center(cp[:,perm]);sh2=dot(shuffled,shuffled);assert not bool(((c2>0)&(sh2==0)).any())
        scale=torch.sqrt(torch.where(sh2>0,c2/torch.where(sh2>0,sh2,torch.ones_like(sh2)),torch.zeros_like(sh2)));shuffled*=scale[:,None]
        shuffle=(F.double()-shuffled).float();shuffle=torch.where((d['alpha']==0)[:,None],F,shuffle)
        pre=dot(correction,s);rounded=dot(F.double()-main.double(),s);rounding=(main.double()-(F.double()-correction)).abs().amax(-1)
        arithmetic=256*torch.finfo(torch.float64).eps*q4.shape[-1]*(1+correction.abs().amax(-1)*s.abs().amax(-1))
        bound=pre.abs()+rounding*(p*s.abs()).sum(-1)+arithmetic
        assert bool((d['axis_unresolved']|(pre.abs()<=arithmetic)).all()) and bool((rounded.abs()<=bound).all())
        # Descriptive fraction of the axis outside numerically resolved span(g,e).
        gbad=g2<=EPS*gc.abs().amax(-1).square();ge=dot(gc,ec);gs=dot(gc,s)
        e1=ec-torch.where(gbad,torch.zeros_like(ge),ge/torch.where(gbad,torch.ones_like(g2),g2))[:,None]*gc
        s1=s-torch.where(gbad,torch.zeros_like(gs),gs/torch.where(gbad,torch.ones_like(g2),g2))[:,None]*gc
        e12=dot(e1,e1);ebad=e12<=threshold;outside=s1-torch.where(ebad,torch.zeros_like(gs),dot(s1,e1)/torch.where(ebad,torch.ones_like(e12),e12))[:,None]*e1
        outside_fraction=torch.where(d['axis2']>0,dot(outside,outside)/torch.where(d['axis2']>0,d['axis2'],torch.ones_like(d['axis2'])),torch.zeros_like(d['axis2']))
        controls=dict(paired_loop_protection=main,reference_only_axis=ref,matched_guidance=matched,shuffled_correction=shuffle,shuffled_axis=axis_shuffle)
        diag=dict(alpha=d['alpha'],old_Fisher_alpha=od['alpha'],protection_b=d['b'],s_norm2=d['axis2'],s_unresolved=d['axis_unresolved'],e_norm2=dot(ec,ec),protected_norm2=dot(d['b'][:,None]*s,d['b'][:,None]*s),
            residual_prior_norm2=dot(d['residual'],d['residual']),guidance_norm2=g2,residual_norm2=r2,correction_norm2=c2,rho=rho,matched_guidance_norm2=rho.square()*g2,shuffled_correction_norm2=dot(shuffled,shuffled),
            orthogonality_pre=pre,orthogonality_rounded=rounded,orthogonality_bound=bound,orthogonality_arithmetic_bound=arithmetic,orthogonality_defined=~d['axis_unresolved'],max_logit_rounding=rounding,
            reference_axis_alpha=rd['alpha'],reference_axis_unresolved=rd['axis_unresolved'],reference_axis_protection_b=rd['b'],reference_axis_correction_norm2=dot(center(rd['correction']),center(rd['correction'])),
            shuffled_axis_alpha=ad['alpha'],shuffled_axis_unresolved=ad['axis_unresolved'],shuffled_axis_correction_norm2=dot(center(ad['correction']),center(ad['correction'])),real_last_loop_norm2=dot(u,u),reference_last_loop_norm2=dot(v,v),axis_outside_ge_span_fraction=outside_fraction)
        assert all(bool(torch.isfinite(x).all()) for x in list(controls.values())+list(diag.values()))
        return controls,diag


def synthetic_checks():
    import torch
    rng=np.random.default_rng(2021);q=[(rng.integers(-20,21,(5,19))/8).astype('float32') for _ in range(3)];r=[(rng.integers(-16,17,(1,5,19))/8).astype('float32') for _ in range(3)];perm=rng.permutation(19)
    tensors=[torch.from_numpy(x) for x in q+r];out,d=torch_logits(*tensors,permutation=perm);z,nd=numpy_primary(*(q+r));ae=float(np.max(abs(nd['alpha']-d['alpha'].numpy())));ze=float(np.max(abs(z-out['paired_loop_protection'].numpy())));assert ae<1e-12 and ze<2e-6
    zero,_=torch_logits(*tensors,permutation=perm,force_zero=True);F=tensors[2]+.5*(tensors[2]-tensors[0]);assert all(torch.equal(x,F) for x in zero.values())
    # Both last-loop transitions set to zero -> main exactly original M2.
    none,dd=torch_logits(tensors[0],tensors[2],tensors[2],tensors[3],tensors[5],tensors[5],permutation=perm)
    from r7_math import token_logits
    old,_=token_logits(tensors[0],tensors[2],tensors[3],tensors[5]);assert torch.equal(none['paired_loop_protection'],old)
    flip,_=torch_logits(tensors[0],2*tensors[2]-tensors[1],tensors[2],tensors[3],2*tensors[5]-tensors[4],tensors[5],permutation=perm);assert torch.equal(flip['paired_loop_protection'],out['paired_loop_protection'])
    for a,b in [('matched_guidance_norm2','residual_norm2'),('shuffled_correction_norm2','correction_norm2')]:assert float((d[a]-d[b]).abs().max())<1e-12
    gauge,gd=torch_logits(tensors[0]+4,tensors[1]+4,tensors[2]+4,tensors[3]+2,tensors[4]+2,tensors[5]+2,permutation=perm);assert float((gd['alpha']-d['alpha']).abs().max())<1e-12
    assert float((gauge['paired_loop_protection'].log_softmax(-1)-out['paired_loop_protection'].log_softmax(-1)).abs().max())<2e-6
    dup,_=torch_logits(*tensors[:3],*(x.expand(3,-1,-1) for x in tensors[3:]),permutation=perm);assert torch.equal(dup['paired_loop_protection'],out['paired_loop_protection'])
    ext=[torch.cat([x,x[:1]]) if x.ndim==2 else torch.cat([x,x[:,:1]],dim=1) for x in tensors];extended,_=torch_logits(*ext,permutation=perm);assert torch.equal(extended['paired_loop_protection'][:-1],out['paired_loop_protection'])
    return dict(status='passed_last_loop_protection_synthetic_checks',NumPy_Torch_alpha_error=ae,FP32_output_error=ze,zero_exact_Fixed=True,unresolved_axis_exact_M2=True,axis_sign_invariant=True,
        norm_controls_checked=True,gauge_checked=True,identical_references_exact=True,batch_extension_exact=True,orthogonality_rounding_bound_checked=True,interpretation='A reference-only falsification is required; no axis semantic/correctness guarantee.')


if __name__=='__main__':
    import json
    print(json.dumps(synthetic_checks(),indent=2))
