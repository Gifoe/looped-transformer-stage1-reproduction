"""Four fixed R6 candidate-score rules; NumPy FP64, no gold/data/model access.

Inputs contain already character-normalized complete-answer likelihoods.
Projection is over the current answer-choice vector, not vocabulary logits.
Clipping produces constrained component removal, not necessarily orthogonality.
"""
import numpy as np

METHOD_IDS=('r6_prior_quarter','r6_prior_half','r6_prior_projection','r6_matched_magnitude')


def _finite(*values):
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError('Nonfinite R6 input, intermediate or output')


def _scores(methods,name):
    try:raw=np.asarray(methods[name]['normalized_scores'])
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('Missing or invalid normalized_scores: '+name) from exc
    if raw.ndim!=1 or raw.size<2 or raw.dtype.kind not in 'iuf':
        raise ValueError('Scores must be a real numeric 1D vector of at least two choices')
    value=raw.astype(np.float64,copy=True);_finite(value)
    return value


def _inputs(original_methods,prior_methods):
    fixed=_scores(original_methods,'loopcd_k4_fixed')
    k4=_scores(original_methods,'native_k4');k5=_scores(original_methods,'native_k5')
    prior_fixed=_scores(prior_methods,'loopcd_k4_fixed');prior_k4=_scores(prior_methods,'native_k4')
    if any(v.shape!=fixed.shape for v in (k4,k5,prior_fixed,prior_k4)):
        raise ValueError('All score vectors must use the same ordered choices')
    with np.errstate(over='ignore',invalid='ignore'):
        incumbent=fixed+(k4-k5);guidance=fixed-k4;prior=prior_fixed-prior_k4
    _finite(incumbent,guidance,prior)
    return incumbent,guidance,prior


def projection_parameters(guidance,prior):
    """Return alpha, rho and centered g/e for synthetic diagnostics.

    alpha minimizes ||center(g)-a center(e)||^2 over a in [0,1].
    rho matches that residual's Euclidean norm; it is not a KL/Fisher norm.
    """
    g=np.asarray(guidance);e=np.asarray(prior)
    if (g.ndim!=1 or g.size<2 or g.shape!=e.shape
        or g.dtype.kind not in 'iuf' or e.dtype.kind not in 'iuf'):
        raise ValueError('Projection requires two matching real choice vectors')
    g=g.astype(np.float64,copy=True);e=e.astype(np.float64,copy=True);_finite(g,e)
    with np.errstate(over='ignore',invalid='ignore'):
        gc=g-g.mean();ec=e-e.mean()
        denominator=np.dot(ec,ec);numerator=np.dot(gc,ec);g2=np.dot(gc,gc)
    _finite(gc,ec,denominator,numerator,g2)
    if denominator==0.:alpha=0.
    else:
        # Avoid overflowing numerator/denominator before clipping to [0,1].
        alpha=0. if numerator<=0. else (1. if numerator>=denominator else float(numerator/denominator))
    if g2==0. or alpha==0.:rho=1.
    else:
        with np.errstate(over='ignore',invalid='ignore'):
            residual=gc-alpha*ec;residual2=np.dot(residual,residual)
        _finite(residual,residual2)
        rho=float(np.sqrt(np.clip(residual2/g2,0.,1.)))
    return alpha,rho,gc,ec


def selected_score(original_methods,prior_methods,method_id):
    """Compute only the selected rule; return detached, owned FP64 ranking scores.

    No labels, predicted answers, frequencies, or other metadata are consulted.
    Inputs are already normalized; no further length normalization occurs here.
    """
    if not isinstance(method_id,str) or method_id not in METHOD_IDS:
        raise ValueError('Unknown R6 method ID')
    incumbent,guidance,prior=_inputs(original_methods,prior_methods)
    with np.errstate(over='ignore',invalid='ignore'):
        if method_id=='r6_prior_quarter':result=incumbent-.25*prior
        elif method_id=='r6_prior_half':result=incumbent-.5*prior
        else:
            alpha,rho,_,_=projection_parameters(guidance,prior)
            if method_id=='r6_prior_projection':
                result=incumbent.copy() if alpha==0. else incumbent-alpha*prior
            else:result=incumbent.copy() if rho==1. else incumbent+(rho-1.)*guidance
    _finite(result)
    return result


def all_scores(original_methods,prior_methods):
    """Training convenience only. Heldout consumers should call selected_score."""
    return {mid:selected_score(original_methods,prior_methods,mid) for mid in METHOD_IDS}
