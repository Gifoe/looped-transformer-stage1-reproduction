"""Question-conditioned protection of prior components: hypothesis, not semantics proof.

No direct joint-score fusion, labels, fitting, weights, gates or model calls.
Robust protection minimizes mean_j ||E-b*S_j||^2 over real b using current
unlabelled forwards. It attenuates protection when reference directions vary.
"""
import numpy as np
from r8_scores import center,norm,finite,EPS
from slcd_r6_scores import _inputs,projection_parameters,selected_score

ELIGIBLE=('single_loop_axis','robust_cyclic_loop_axis')


def axis_rows(aux,kind,multi):
    refs=aux['cyclic_rotations'] if multi else [dict(sum_scores=aux['auxiliary_sum_scores'])]
    vectors=[];scale=0.
    for ref in refs:
        s=ref['sum_scores'];q4=center(s['native_k4']);q1=center(s['native_k1'])
        vectors.append(center(q4-q1) if kind=='loop' else q4)
        scale=max(scale,norm(q4),norm(q1) if kind=='loop' else 0.)
    return np.asarray(vectors),scale


def repair(a,g,e,r6,semantic,reference_scale):
    a,g,e,r6=[np.asarray(v,dtype=np.float64) for v in [a,g,e,r6]]
    ss=np.asarray(semantic,dtype=np.float64)
    assert a.ndim==1 and a.size>=2 and all(v.shape==a.shape for v in [g,e,r6]) and ss.ndim==2 and ss.shape[1]==len(a)
    finite(a,g,e,r6,ss,reference_scale)
    ec=center(e);gc=center(g);en=norm(ec);sn=max(norm(s) for s in ss)
    if sn==0. or sn<=EPS*reference_scale:
        return r6.copy(),dict(semantic_unresolved=True,prior_residual_zero=False,protected_prior_norm=0.,
            raw_prior_norm=en,protection_consensus=0.,orthogonality_guaranteed=False)
    normalized=ss/sn;mean=normalized.mean(axis=0);den=float(np.mean(np.sum(normalized**2,axis=1)))
    assert den>0.
    coefficient=float(ec@mean)/den;protected=coefficient*mean
    consensus=float(mean@mean)/den
    assert -1e-12<=consensus<=1.+1e-12
    v=center(ec-protected);vn=norm(v)
    residual_zero=vn<=EPS*en
    if residual_zero:alpha=0.;out=a.copy();v=np.zeros_like(v)
    else:
        alpha,_,_,_=projection_parameters(g,e-protected)
        out=a-alpha*(e-protected)
    assert norm(v)<=en+1e-12*max(en,1.)
    finite(out,protected,alpha,consensus)
    diag=dict(semantic_unresolved=False,prior_residual_zero=bool(residual_zero),alpha=float(alpha),raw_prior_norm=en,
        protected_prior_norm=norm(protected),residual_prior_norm=norm(v),protection_consensus=consensus,
        removed_component=protected.tolist(),reference_count=len(ss),orthogonality_guaranteed=len(ss)==1,
        residual_dot_mean_axis=float(v@mean),correction_centered_norm=norm(center(out-a)),
        scope=('Single reference preserves A1 centered component along the auxiliary axis; no correctness or R6 margin guarantee.' if len(ss)==1 else
               'Robust references attenuate prior removal along a consensus auxiliary axis; no exact component, correctness or R6 margin guarantee.'))
    return out,diag


def all_scores(original,prior,aux):
    a,g,e=_inputs(original,prior);r=selected_score(original,prior,'r6_prior_projection');values={};diagnostics={}
    for name,kind,multi in [('single_loop_axis','loop',False),('robust_cyclic_loop_axis','loop',True),
                            ('single_native_axis','native',False),('robust_cyclic_native_axis','native',True)]:
        axes,scale=axis_rows(aux,kind,multi);out,d=repair(a,g,e,r,axes,scale);values[name]=out;diagnostics[name]=d
        if name in ELIGIBLE:
            delta=center(out-a);ec=center(e);en=norm(ec);size=norm(delta)
            values[name+'_prior_norm']=a.copy() if en==0. else a-(size/en)*e
            p=np.random.default_rng(2021).permutation(len(a));values[name+'_shuffle_update']=a+delta[p]
            d.update(prior_norm_control_error=abs(norm(center(values[name+'_prior_norm']-a))-size),
                shuffled_update_norm_error=abs(norm(center(values[name+'_shuffle_update']-a))-size),
                shuffle_kind='Permutes completed centered correction, exact correction-norm match; not a recomputed reference projection',
                original_R6_correction_norm=norm(center(r-a)))
    finite(*values.values());return values,diagnostics
