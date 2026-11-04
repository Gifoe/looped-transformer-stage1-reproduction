"""R7 label-free choice/vocabulary projection; no data/model/target access.

q1/q4: FP32 [..., V]; refs1/refs4: FP32 [J, ..., V], J>=1.
g/e/Fixed arithmetic and returned logits are FP32. The softmax measure p and
ALL projection statistics are FP64 over the entire vocabulary. Final FP32
log_softmax, target gathering, candidate accumulation and ONE character-length
normalization belong to the caller. Add Q4-Q5 there; never subtract R6 again.

Modes: robust, mean_only, bridge (J=1 + externally frozen whole_alpha),
matched_scale (robust coefficient), shuffle (robust correction + supplied fixed
permutation). Shuffle is a control, not an eligible model. No random sampling.
Indistinguishable D <= 64*eps64*mean_j(max_v(abs(ec_j))**2) gives alpha=0.
This rule applies to token estimators; bridge explicitly uses external alpha.
M1 uses R6's exact zero rule instead, to preserve exact J=1 compatibility.
Diagnostics are per-position backend arrays, not target-conditioned estimates.
projected_correction_norm2 is ||alpha*ebar||_p^2 before control transforms;
applied_direction_norm2 is the actual guidance displacement's squared norm
(for matched_scale, ||(1-rho)*gc||_p^2), before final FP32 rounding.
"""
import numpy as np

MODES=('robust','mean_only','bridge','matched_scale','shuffle')
EPS_MULTIPLIER=64.


def _finite(*values):
    if any(not np.isfinite(x).all() for x in values):
        raise ValueError('Nonfinite input, intermediate or output')


def candidate_scores(original_methods, prior_methods_list):
    """M1: already character-normalized scores; never consult gold metadata.

Returns (owned float64 choice vector, {alpha,numerator,denominator,n_references,
single_reference_exact_path}). All references use the same ordered choices.
Only an entirely identical reference collection collapses to its J=1 path.
"""
    from slcd_r6_scores import _inputs, projection_parameters, selected_score
    if not isinstance(prior_methods_list,(tuple,list)) or not prior_methods_list:
        raise ValueError('At least one reference is required')
    a,g,e0=_inputs(original_methods,prior_methods_list[0]);es=[e0]
    for prior in prior_methods_list[1:]:
        _,_,e=_inputs(original_methods,prior);es.append(e)
    same=all(np.array_equal(e,e0) for e in es)
    if same:
        alpha,_,gc,ec=projection_parameters(g,e0)
        output=selected_score(original_methods,prior_methods_list[0],'r6_prior_projection')
        num=float(gc@ec);den=float(ec@ec)
    else:
        with np.errstate(over='ignore',invalid='ignore'):
            es=np.stack(es);gc=g-g.mean();ec=es-es.mean(axis=-1,keepdims=True)
            ebar=ec.mean(axis=0);num=float(gc@ebar);den=float(np.mean(np.sum(ec*ec,axis=-1)))
            _finite(gc,ec,ebar,num,den)
            alpha=0. if den==0 or num<=0 else 1. if num>=den else num/den
            output=a.copy() if alpha==0 else a-alpha*es.mean(axis=0)
    _finite(output,num,den)
    return output,dict(alpha=alpha,numerator=num,denominator=den,
                       n_references=len(prior_methods_list),single_reference_exact_path=same)


def _np_inputs(q1,q4,refs1,refs4):
    arrays=[np.asarray(x) for x in (q1,q4,refs1,refs4)]
    if any(x.dtype!=np.float32 for x in arrays):raise ValueError('Logits must be FP32')
    a,b,c,d=arrays
    if (a.ndim<1 or a.shape!=b.shape or a.shape[-1]<2 or not a.size
        or c.ndim!=a.ndim+1 or c.shape!=d.shape or c.shape[0]<1 or c.shape[1:]!=a.shape):
        raise ValueError('Expected q [...,V], references [J,...,V], V>=2')
    _finite(*arrays);return arrays


def _np_center(x,p):
    shifted=x-x[..., :1]
    return shifted-np.sum(p*shifted,axis=-1,keepdims=True)


def token_logits_numpy(q1,q4,refs1,refs4,mode='robust',whole_alpha=None,
                       permutation=None,force_zero=False):
    """Independent NumPy reference implementation; see module contract."""
    if mode not in MODES:raise ValueError('Unknown projection mode')
    q1,q4,refs1,refs4=_np_inputs(q1,q4,refs1,refs4)
    j=refs1.shape[0]
    if mode!='bridge' and whole_alpha is not None:raise ValueError('whole_alpha is bridge-only')
    if mode!='shuffle' and permutation is not None:raise ValueError('permutation is shuffle-only')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        g=np.float32(.5)*(q4-q1);e=np.float32(.5)*(refs4-refs1);fixed=q4+g
        z=q4.astype(np.float64);p=np.exp(z-z.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
        gc=_np_center(g.astype(np.float64),p);ec=_np_center(e.astype(np.float64),p[None])
        if np.array_equal(ec,np.broadcast_to(ec[:1],ec.shape)):ec=ec[:1]
        ebar=ec.mean(axis=0);robust=np.mean(np.sum(p[None]*ec*ec,axis=-1),axis=0)
        meanvar=np.sum(p*ebar*ebar,axis=-1);num=np.sum(p*gc*ebar,axis=-1)
        threshold=EPS_MULTIPLIER*np.finfo(np.float64).eps*np.mean(np.max(np.abs(ec),axis=-1)**2,axis=0)
        den=meanvar if mode=='mean_only' else robust
        bad=den<=threshold;alpha=np.zeros_like(den)
        np.divide(num,den,out=alpha,where=~bad);alpha=np.clip(alpha,0.,1.)
        if mode=='bridge':
            if j!=1 or whole_alpha is None:raise ValueError('bridge requires J=1 and whole_alpha')
            alpha=np.broadcast_to(np.asarray(whole_alpha,dtype=np.float64),den.shape).copy()
            _finite(alpha)
            if np.any((alpha<0)|(alpha>1)):raise ValueError('whole_alpha must be in [0,1]')
        if force_zero:alpha=np.zeros_like(alpha)
        correction=alpha[...,None]*ebar;residual=gc-correction
        g2=np.sum(p*gc*gc,axis=-1);r2=np.sum(p*residual*residual,axis=-1)
        rho=np.ones_like(g2);np.divide(r2,g2,out=rho,where=g2>0);rho=np.sqrt(np.maximum(rho,0.))
        correction2=np.sum(p*correction*correction,axis=-1)
        if mode=='matched_scale':
            output=(q4.astype(np.float64)+rho[...,None]*g.astype(np.float64)).astype(np.float32)
        else:
            if mode=='shuffle':
                perm=np.asarray(permutation)
                if perm.shape!=(q4.shape[-1],) or perm.dtype.kind not in 'iu' or not np.array_equal(np.sort(perm),np.arange(q4.shape[-1])):
                    raise ValueError('shuffle requires a full fixed vocabulary permutation')
                shuffled=_np_center(correction[...,perm],p);norm2=np.sum(p*shuffled*shuffled,axis=-1)
                if np.any((correction2>0)&(norm2==0)):raise ValueError('Cannot match shuffled norm on zero support')
                scale=np.zeros_like(norm2);np.divide(correction2,norm2,out=scale,where=norm2>0)
                correction=shuffled*np.sqrt(scale)[...,None]
            output=(fixed.astype(np.float64)-correction).astype(np.float32)
        # Preserve the literal original Fixed path when the applied alpha is zero.
        output=np.where((alpha==0)[...,None],fixed,output).astype(np.float32)
        applied2=(1.-rho)**2*g2 if mode=='matched_scale' else np.sum(p*correction*correction,axis=-1)
    _finite(g,e,fixed,p,gc,ec,ebar,num,den,threshold,alpha,g2,r2,rho,correction,output,applied2)
    return output,dict(alpha=alpha,numerator=num,denominator=den,robust_denominator=robust,
        mean_only_denominator=meanvar,indistinguishable_threshold=threshold,
        indistinguishable=bad,guidance_norm2=g2,residual_norm2=r2,rho=rho,
        projected_correction_norm2=correction2,applied_direction_norm2=applied2,
        n_references=j,mode=mode,force_zero=bool(force_zero))


def token_logits(q1,q4,refs1,refs4,mode='robust',whole_alpha=None,
                 permutation=None,force_zero=False):
    """Torch CPU/GPU version. No gradients, mutation, targets or model access.

Returned tensors are detached; diagnostics stay on the input device. The caller
may reduce/copy diagnostics, but should not persist full-vocabulary activations.
"""
    import torch
    if mode not in MODES:raise ValueError('Unknown projection mode')
    with torch.no_grad():
        values=(q1,q4,refs1,refs4)
        if any(not isinstance(x,torch.Tensor) or x.dtype!=torch.float32 or x.device!=q4.device for x in values):
            raise ValueError('All logits must be FP32 tensors on one device')
        q1,q4,refs1,refs4=[x.detach() for x in values]
        if (q1.ndim<1 or q1.shape!=q4.shape or q1.shape[-1]<2 or q1.numel()==0
            or refs1.ndim!=q1.ndim+1 or refs1.shape!=refs4.shape or refs1.shape[0]<1 or refs1.shape[1:]!=q1.shape):
            raise ValueError('Expected q [...,V], references [J,...,V], V>=2')
        def finite(*xs):
            if any(not bool(torch.isfinite(x).all()) for x in xs):raise ValueError('Nonfinite input, intermediate or output')
        finite(q1,q4,refs1,refs4);j=refs1.shape[0]
        if mode!='bridge' and whole_alpha is not None:raise ValueError('whole_alpha is bridge-only')
        if mode!='shuffle' and permutation is not None:raise ValueError('permutation is shuffle-only')
        def center(x):
            shifted=x-x[..., :1]
            return shifted-(p*shifted).sum(dim=-1,keepdim=True)
        g=.5*(q4-q1);e=.5*(refs4-refs1);fixed=q4+g
        p=torch.softmax(q4.double(),dim=-1);gc=center(g.double());ec=center(e.double())
        if torch.equal(ec,ec[:1].expand_as(ec)):ec=ec[:1]
        ebar=ec.mean(dim=0);robust=(p*ec*ec).sum(dim=-1).mean(dim=0)
        meanvar=(p*ebar*ebar).sum(dim=-1);num=(p*gc*ebar).sum(dim=-1)
        threshold=EPS_MULTIPLIER*torch.finfo(torch.float64).eps*ec.abs().amax(dim=-1).square().mean(dim=0)
        den=meanvar if mode=='mean_only' else robust;bad=den<=threshold
        alpha=(num/torch.where(bad,torch.ones_like(den),den)).clamp(0.,1.)
        alpha=torch.where(bad,torch.zeros_like(alpha),alpha)
        if mode=='bridge':
            if j!=1 or whole_alpha is None:raise ValueError('bridge requires J=1 and whole_alpha')
            alpha=torch.broadcast_to(torch.as_tensor(whole_alpha,dtype=torch.float64,device=q4.device).detach(),den.shape).clone()
            finite(alpha)
            if bool(((alpha<0)|(alpha>1)).any()):raise ValueError('whole_alpha must be in [0,1]')
        if force_zero:alpha=torch.zeros_like(alpha)
        correction=alpha.unsqueeze(-1)*ebar;residual=gc-correction
        g2=(p*gc*gc).sum(dim=-1);r2=(p*residual*residual).sum(dim=-1)
        rho=torch.sqrt(torch.where(g2>0,r2/torch.where(g2>0,g2,torch.ones_like(g2)),torch.ones_like(g2)).clamp_min(0.))
        correction2=(p*correction*correction).sum(dim=-1)
        if mode=='matched_scale':output=(q4.double()+rho.unsqueeze(-1)*g.double()).float()
        else:
            if mode=='shuffle':
                if permutation is None:raise ValueError('shuffle requires a fixed vocabulary permutation')
                perm=torch.as_tensor(permutation,device=q4.device)
                if perm.shape!=(q4.shape[-1],) or perm.dtype not in (torch.int32,torch.int64) or not torch.equal(perm.sort().values,torch.arange(q4.shape[-1],device=q4.device)):
                    raise ValueError('shuffle requires a full fixed vocabulary permutation')
                shuffled=center(correction[...,perm.long()]);norm2=(p*shuffled*shuffled).sum(dim=-1)
                if bool(((correction2>0)&(norm2==0)).any()):raise ValueError('Cannot match shuffled norm on zero support')
                scale=torch.where(norm2>0,correction2/torch.where(norm2>0,norm2,torch.ones_like(norm2)),torch.zeros_like(norm2))
                correction=shuffled*torch.sqrt(scale).unsqueeze(-1)
            output=(fixed.double()-correction).float()
        output=torch.where((alpha==0).unsqueeze(-1),fixed,output)
        applied2=(1.-rho).square()*g2 if mode=='matched_scale' else (p*correction*correction).sum(dim=-1)
        finite(g,e,fixed,p,gc,ec,ebar,num,den,threshold,alpha,g2,r2,rho,correction,output,applied2)
        return output,dict(alpha=alpha,numerator=num,denominator=den,robust_denominator=robust,
            mean_only_denominator=meanvar,indistinguishable_threshold=threshold,indistinguishable=bad,
            guidance_norm2=g2,residual_norm2=r2,rho=rho,projected_correction_norm2=correction2,
            applied_direction_norm2=applied2,n_references=j,mode=mode,force_zero=bool(force_zero))
