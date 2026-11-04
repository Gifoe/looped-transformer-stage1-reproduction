"""One full-vocabulary uniform-metric prior projection, no fitted temperature.

Standard Euclidean projection, not a new principle. Native Fisher effective
rank is diagnostic; it never gates or scales the decoding coefficient.
"""
import numpy as np
EPS=64.*np.finfo(np.float64).eps
RANK_TRACE_THRESHOLD=1e-8


def numpy_primary(q1,q4,r1,r4):
    q1,q4,r1,r4=[np.asarray(x) for x in [q1,q4,r1,r4]]
    assert all(x.dtype==np.float32 for x in [q1,q4,r1,r4]) and r1.shape==r4.shape and r1.shape[1:]==q1.shape==q4.shape
    assert np.array_equal(r1,np.broadcast_to(r1[:1],r1.shape)) and np.array_equal(r4,np.broadcast_to(r4[:1],r4.shape))
    g=.5*(q4-q1);e=.5*(r4[0]-r1[0]);gu=g.astype(float)-g[:,:1].astype(float);gu-=gu.mean(-1,keepdims=True);eu=e.astype(float)-e[:,:1].astype(float);eu-=eu.mean(-1,keepdims=True)
    D=np.mean(eu*eu,axis=-1);N=np.mean(gu*eu,axis=-1);bad=D<=EPS*np.max(np.abs(eu),axis=-1)**2
    a=np.clip(np.divide(N,D,out=np.zeros_like(D),where=~bad),0.,1.);F=q4+g;out=(F.astype(float)-a[:,None]*eu).astype('float32');out=np.where((a==0)[:,None],F,out)
    return out,a


def torch_logits(q1,q4,refs1,refs4,permutation,force_zero=False):
    import torch
    from r7_math import token_logits
    assert q1.shape==q4.shape and q1.ndim==2 and refs1.shape==refs4.shape and refs1.shape[1:]==q1.shape
    assert all(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) for x in [q1,q4,refs1,refs4])
    assert torch.equal(refs1,refs1[:1].expand_as(refs1)) and torch.equal(refs4,refs4[:1].expand_as(refs4)),'Only one reference or identical duplicates supported in this pilot'
    with torch.no_grad():
        g=.5*(q4-q1);e=.5*(refs4[0]-refs1[0]);F=q4+g;p=q4.double().softmax(-1)
        def centerU(x):
            y=x-x[:,:1];return y-y.mean(-1,keepdim=True)
        def centerP(x):
            y=x-x[:,:1];return y-(p*y).sum(-1,keepdim=True)
        def U2(x):return x.square().mean(-1)
        def P2(x):return (p*x.square()).sum(-1)
        gu,eu=centerU(g.double()),centerU(e.double());gp,ep=centerP(g.double()),centerP(e.double());den=U2(eu);num=(gu*eu).mean(-1);threshold=EPS*eu.abs().amax(-1).square();bad=den<=threshold
        a=(num/torch.where(bad,torch.ones_like(den),den)).clamp(0.,1.);a=torch.where(bad,torch.zeros_like(a),a)
        if force_zero:a=torch.zeros_like(a)
        correction=a[:,None]*eu;cp=centerP(correction);gU2=U2(gu);rU2=U2(gu-correction);cU2=U2(correction);gP2=P2(gp);rP2=P2(gp-cp);cP2=P2(cp)
        def ratio(n,d):return torch.sqrt(torch.where(d>0,n/torch.where(d>0,d,torch.ones_like(d)),torch.ones_like(d)))
        rhoU,rhoP=ratio(rU2,gU2),ratio(rP2,gP2)
        assert not bool(((gU2==0)&(rU2>threshold)).any()) and not bool(((gP2==0)&(rP2>EPS*gp.abs().amax(-1).square())).any())
        perm=torch.as_tensor(permutation,device=q4.device);assert perm.shape==(q4.shape[-1],) and torch.equal(perm.sort().values,torch.arange(q4.shape[-1],device=q4.device))
        su=centerU(correction[:,perm]);sp=centerP(cp[:,perm]);su2,sp2=U2(su),P2(sp)
        assert not bool(((cU2>0)&(su2==0)).any()) and not bool(((cP2>0)&(sp2==0)).any())
        su=su*torch.sqrt(torch.where(su2>0,cU2/torch.where(su2>0,su2,torch.ones_like(su2)),torch.zeros_like(su2)))[:,None]
        sp=sp*torch.sqrt(torch.where(sp2>0,cP2/torch.where(sp2>0,sp2,torch.ones_like(sp2)),torch.zeros_like(sp2)))[:,None]
        raw=dict(uniform_projection=F.double()-correction,matched_guidance_U=q4.double()+rhoU[:,None]*g.double(),matched_guidance_P=q4.double()+rhoP[:,None]*g.double(),shuffled_correction_U=F.double()-su,shuffled_correction_P=F.double()-sp)
        controls={k:torch.where((a==0)[:,None],F,z.float()) for k,z in raw.items()}
        old,od=token_logits(q1,q4,refs1,refs4)
        # Stable diagonal/off-diagonal sum for tr((diag(p)-ppT)^2).
        p2=p.square();s2=p2.sum(-1);top=p.argmax(-1,keepdim=True);rest=p2.clone();rest.scatter_(-1,top,0.);other=s2[:,None]-p2;other.scatter_(-1,top,rest.sum(-1,keepdim=True))
        trace=(p*(1-p)).sum(-1);trace2=(p2*((1-p).square()+other)).sum(-1);rank_defined=(trace>RANK_TRACE_THRESHOLD)&(trace2>0)
        rank=torch.where(rank_defined,trace.square()/torch.where(rank_defined,trace2,torch.ones_like(trace2)),torch.zeros_like(trace))
        def cosine(x,y,metric):
            x2,y2=(U2(x),U2(y)) if metric=='U' else (P2(x),P2(y));inner=(x*y).mean(-1) if metric=='U' else (p*x*y).sum(-1)
            ok=(x2>EPS*x.abs().amax(-1).square())&(y2>EPS*y.abs().amax(-1).square());d=torch.sqrt(x2*y2)
            return torch.where(ok,inner/torch.where(ok,d,torch.ones_like(d)),torch.zeros_like(d)),ok
        cosU,okU=cosine(gu,eu,'U');cosP,okP=cosine(gp,ep,'P');shuffleU,oksU=cosine(gu,centerU(eu[:,perm]),'U');shuffleP,oksP=cosine(gp,centerP(ep[:,perm]),'P')
        diag=dict(alpha=a,old_Fisher_alpha=od['alpha'],uniform_numerator=num,uniform_denominator=den,uniform_threshold=threshold,
            guidance_U_norm2=gU2,residual_U_norm2=rU2,correction_U_norm2=cU2,guidance_P_norm2=gP2,residual_P_norm2=rP2,correction_P_norm2=cP2,rho_U=rhoU,rho_P=rhoP,
            matched_U_norm2=rhoU.square()*gU2,matched_P_norm2=rhoP.square()*gP2,shuffle_U_norm2=U2(su),shuffle_P_norm2=P2(sp),
            old_Fisher_correction_P_norm2=od['projected_correction_norm2'],old_Fisher_correction_U_norm2=U2(centerU(od['alpha'][:,None]*ep)),
            Fisher_effective_rank=rank,Fisher_rank_defined=rank_defined,Fisher_trace=trace,Fisher_trace_square=trace2,native_probability_IPR=1/s2,pmax=p.amax(-1),native_entropy=-(p*q4.double().log_softmax(-1)).sum(-1),
            uniform_alignment=cosU,uniform_alignment_defined=okU,native_alignment=cosP,native_alignment_defined=okP,uniform_shuffle_alignment=shuffleU,uniform_shuffle_alignment_defined=oksU,native_shuffle_alignment=shuffleP,native_shuffle_alignment_defined=oksP)
        assert all(bool(torch.isfinite(x).all()) for x in controls.values()) and all(bool(torch.isfinite(x).all()) for x in diag.values())
        return controls,diag


def synthetic_checks():
    import torch
    rng=np.random.default_rng(2021);q=(rng.integers(-20,21,(5,19))/8).astype('float32');q1=q+(rng.integers(-8,9,q.shape)/8).astype('float32');r4=(rng.integers(-16,17,(1,)+q.shape)/8).astype('float32');r1=r4+(rng.integers(-8,9,r4.shape)/8).astype('float32');perm=rng.permutation(19)
    tensors=[torch.from_numpy(x) for x in [q1,q,r1,r4]];out,d=torch_logits(*tensors,permutation=perm);z,a=numpy_primary(q1,q,r1,r4)
    ae=float(np.max(abs(a-d['alpha'].numpy())));ze=float(np.max(abs(z-out['uniform_projection'].numpy())));assert ae<1e-12 and ze<2e-6
    for x,y in [('matched_U_norm2','residual_U_norm2'),('matched_P_norm2','residual_P_norm2'),('shuffle_U_norm2','correction_U_norm2'),('shuffle_P_norm2','correction_P_norm2')]:assert float((d[x]-d[y]).abs().max())<1e-12
    zero,_=torch_logits(*tensors,permutation=perm,force_zero=True);F=torch.from_numpy(q+.5*(q-q1));assert all(torch.equal(x,F) for x in zero.values())
    du,_=torch_logits(tensors[0],tensors[1],tensors[2].expand(3,-1,-1),tensors[3].expand(3,-1,-1),permutation=perm);assert torch.equal(du['uniform_projection'],out['uniform_projection'])
    gauge,gd=torch_logits(tensors[0]+4,tensors[1]+4,tensors[2]+2,tensors[3]+2,permutation=perm);assert float((gd['alpha']-d['alpha']).abs().max())<1e-12
    assert float((gauge['uniform_projection'].log_softmax(-1)-out['uniform_projection'].log_softmax(-1)).abs().max())<2e-6
    extend,xd=torch_logits(torch.cat([tensors[0],tensors[0][:1]]),torch.cat([tensors[1],tensors[1][:1]]),torch.cat([tensors[2],tensors[2][:,:1]],dim=1),torch.cat([tensors[3],tensors[3][:,:1]],dim=1),permutation=perm)
    assert torch.equal(extend['uniform_projection'][:-1],out['uniform_projection'])
    # An exact local Fisher correlation need not imply globally shared direction.
    qq=np.array([[0,0,-80,-80]],dtype='float32');g=np.array([[1,-1,1,-1]],dtype='float32');e=np.array([[1,-1,-1,1]],dtype='float32')
    co,cd=torch_logits(*(torch.from_numpy(x) for x in [qq-2*g,qq,(-2*e)[None],np.zeros((1,1,4),dtype='float32')]),permutation=np.arange(4))
    assert cd['alpha'].item()==0 and abs(cd['old_Fisher_alpha'].item()-1)<1e-12 and abs(cd['Fisher_effective_rank'].item()-1)<1e-12
    # Check stable rank against explicit covariance matrix on small synthetic V.
    p=torch.from_numpy(q).double().softmax(-1)
    H=torch.diag_embed(p)-p[:,:,None]*p[:,None,:];explicit=H.diagonal(dim1=-2,dim2=-1).sum(-1).square()/H.square().sum((-1,-2));assert float((explicit-d['Fisher_effective_rank']).abs().max())<1e-11
    return dict(status='passed_uniform_projection_synthetic_checks',NumPy_Torch_alpha_error=ae,FP32_output_error=ze,zero_exact_Fixed=True,identical_references_exact=True,both_metric_controls_checked=True,
        rank_explicit_matrix_checked=True,two_support_witness=dict(native_Fisher_alpha=cd['old_Fisher_alpha'].item(),uniform_alpha=cd['alpha'].item(),Fisher_effective_rank=cd['Fisher_effective_rank'].item()),
        interpretation='A standard fixed metric ablation, not a proof that low Fisher rank caused accuracy loss; rank never controls decoding.')


if __name__=='__main__':
    import json
    print(json.dumps(synthetic_checks(),indent=2))
