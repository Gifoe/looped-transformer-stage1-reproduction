"""FP32 logit readouts on original-model trajectories; no fitting/model loading.

Source: https://arxiv.org/html/2610.02185v1#S5.SS2 and #A6.SS2.
Those sections specify parallel/orthogonal components relative to final logits,
but do NOT explicitly specify centering. centered=True is our documented
softmax-shift-invariant convention, not an asserted exact author implementation.
centered=False is the literal raw-vector projection control.

Recipes:
 fixed: z_f + omega*(z_f-z_ref)
 adaptive: z_f + omega*(1 - top2_probability_gap(z_f))*(z_f-z_ref)
 two_sided_logit: fixed + gamma*(z_f-z_late)  [subtract late update]
 orthogonal: z_f + omega*orthogonal(z_f-z_ref, z_f)
 parallel: z_f + omega*parallel(z_f-z_ref, z_f)

Adaptive omega is its CAP (default1); fixed/projection omega defaults0.5.
Gamma applies ONLY to two_sided_logit and defaults0. A gamma0 early return
reuses the exact Fixed arithmetic; no late logits are required in that case.
Different reference/final indices and late5/6 are supported explicitly; this
module supplies no search loop, no data access and no default candidate grid.
"""
import math
from numbers import Integral
import torch


def _number(value, name, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int,float)):
        raise ValueError(name+' must be a finite numeric scalar')
    value=float(value)
    if not math.isfinite(value) or value<0 or (upper is not None and value>upper):
        raise ValueError(name+' outside its supported bounds')
    return value


def _index(value, name):
    if isinstance(value,bool) or not isinstance(value,Integral) or not 1<=int(value)<=6:
        raise ValueError(name+' must be an integer recurrent depth in1..6')
    return int(value)


def _matrix(value, name):
    if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.ndim!=2:
        raise ValueError(name+' must be FP32 [target_prediction_position,vocabulary]')
    if value.shape[0]==0 or value.shape[1]<2 or not torch.isfinite(value).all():
        raise ValueError(name+' must be nonempty and finite with vocabulary>=2')
    if value.requires_grad:
        raise ValueError('Detach forward logits; this readout never builds an autograd graph')


@torch.no_grad()
def decompose_contrast(final, reference, *, centered=True):
    """Return (parallel, orthogonal) for each position independently.

    Center both z_final and z_final-z_reference over vocabulary when requested.
    p=a*sum(d*a)/sum(a*a), o=d-p. For EXACTLY zero norm(a), set p=0, o=d.
    No positive epsilon ridge is inserted: that would alter the projection.
    FP32 operations match inference readouts; finite checks reject overflow.
    Inputs are never modified. Components sum to centered contrast when centered.
    """
    _matrix(final,'final');_matrix(reference,'reference')
    if final.shape!=reference.shape or final.device!=reference.device:
        raise ValueError('Projection inputs must have identical shape/device')
    if not isinstance(centered,bool):raise ValueError('centered must be an explicit Boolean')
    d=final-reference
    a=final
    if centered:
        a=final-final.mean(dim=-1,keepdim=True)
        d=d-d.mean(dim=-1,keepdim=True)
    numerator=(d*a).sum(dim=-1,keepdim=True)
    denominator=a.square().sum(dim=-1,keepdim=True)
    if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all():
        raise ValueError('FP32 projection overflow')
    safe=torch.where(denominator>0,denominator,torch.ones_like(denominator))
    coefficient=torch.where(denominator>0,numerator/safe,torch.zeros_like(numerator))
    parallel=coefficient*a
    orthogonal=d-parallel
    if not torch.isfinite(parallel).all() or not torch.isfinite(orthogonal).all():
        raise ValueError('Nonfinite projected component')
    return parallel,orthogonal


@torch.no_grad()
def recipe_logits(logits, recipe):
    """Build one full-vocabulary logit matrix. Does not mutate caller tensors."""
    kind=recipe['kind']
    if kind not in {'fixed','adaptive','two_sided_logit','orthogonal','parallel'}:
        raise ValueError('Unsupported recipe kind: '+str(kind))
    final_index=_index(recipe.get('final',4),'final')
    reference_index=_index(recipe.get('reference',1),'reference')
    if reference_index>=final_index:raise ValueError('Require reference < final')
    final=logits[final_index];reference=logits[reference_index]
    _matrix(final,'final');_matrix(reference,'reference')
    if final.shape!=reference.shape or final.device!=reference.device:
        raise ValueError('Logits must share candidate target positions/vocabulary/device')
    omega=_number(recipe.get('omega',1.0 if kind=='adaptive' else .5),'omega',upper=1.0)
    gamma=_number(recipe.get('gamma',0.),'gamma',upper=1.0)
    if kind!='two_sided_logit' and gamma!=0:
        raise ValueError('gamma is only defined for two_sided_logit')
    if kind in {'orthogonal','parallel'}:
        parallel,orthogonal=decompose_contrast(final,reference,centered=recipe.get('centered',True))
        result=final+omega*(orthogonal if kind=='orthogonal' else parallel)
    elif kind=='adaptive':
        top=final.softmax(-1).topk(2,dim=-1).values
        strength=omega*(1-(top[:,0]-top[:,1]))
        result=final+strength[:,None]*(final-reference)
    else:
        result=final+omega*(final-reference)
        if kind=='two_sided_logit' and gamma!=0:
            late_index=_index(recipe.get('late',5),'late')
            if late_index<=final_index:raise ValueError('Require late > final')
            late=logits[late_index];_matrix(late,'late')
            if late.shape!=final.shape or late.device!=final.device:
                raise ValueError('Late logits must share final shape/device')
            result=result+gamma*(final-late)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite recipe logits')
    return result


@torch.no_grad()
def score_logit_recipes(logits, targets, recipes):
    """Return {recipe_id: full-continuation summed log likelihood}.

    logits[k] is FP32 cast from BF16 head output, [ntokens,full_vocab]. The caller
    supplies ALL target-prediction positions, including every continuation token.
    This function performs full-vocabulary log_softmax BEFORE gather and FP32 sum.
    Candidate/character normalization belongs to the caller, after this function.
    Recipes are processed sequentially to bound temporary vocabulary memory.
    """
    recipes=list(recipes)
    ids=[r.get('id') for r in recipes]
    if any(not isinstance(i,str) or not i for i in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Recipe IDs must be nonempty unique strings')
    if not recipes:return {}
    target=torch.as_tensor(targets)
    if target.ndim!=1 or target.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
        raise ValueError('Targets must be a1D integer sequence')
    output={}
    for r in recipes:
        z=recipe_logits(logits,r)
        if target.numel()!=z.shape[0] or target.numel()==0:
            raise ValueError('Exactly one target for every supplied prediction position required')
        t=target.to(device=z.device,dtype=torch.long)
        if int(t.min())<0 or int(t.max())>=z.shape[1]:raise ValueError('Target outside full vocabulary')
        token_lp=z.log_softmax(-1).gather(-1,t[:,None]).squeeze(-1)
        if not torch.isfinite(token_lp).all():raise ValueError('Nonfinite target likelihood')
        output[r['id']]=float(token_lp.sum())
    return output
