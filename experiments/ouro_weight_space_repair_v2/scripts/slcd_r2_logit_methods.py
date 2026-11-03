"""Six frozen, training-free R2 logit readouts; no model/data loading.

The Fisher inner product is Cov_softmax(z4), including its rank-one term.
Centering is used ONLY inside inner products; returned updates use raw d/u.
Clipped coefficients produce bounded decorrelation, NOT exact orthogonality.
All arithmetic is FP32, on detached logits from the original BF16 head.
"""
import math
import torch
import slcd_logit_methods as original

KINDS = {'fisher_signed', 'fisher_positive', 'euclidean_signed',
         'fisher_matched_norm', 'fisher_cosine', 'ref2_late'}


def _finite(*values):
    if any(not torch.isfinite(v).all() for v in values):
        raise ValueError('Nonfinite R2 intermediate or output')


@torch.no_grad()
def geometry(final, d, u, *, uniform=False):
    """Per-position cov(d,u), var(d), var(u), using stable constant removal.

Subtracting the first coordinate before centering makes exactly constant
vectors have exactly zero variance, even if sum(softmax) rounds away from one.
No epsilon ridge or probability truncation changes the stated geometry.
"""
    for name, value in [('final',final),('d',d),('u',u)]:
        original._matrix(value,name)
        if value.shape != final.shape or value.device != final.device:
            raise ValueError('R2 geometry requires identical shapes/devices')
    dd=d-d[:,:1]; uu=u-u[:,:1]
    if uniform:
        dc=dd-dd.mean(-1,keepdim=True); uc=uu-uu.mean(-1,keepdim=True)
        cov=(dc*uc).mean(-1,keepdim=True)
        vd=dc.square().mean(-1,keepdim=True)
        vu=uc.square().mean(-1,keepdim=True)
    else:
        p=final.softmax(-1)
        dc=dd-(p*dd).sum(-1,keepdim=True)
        uc=uu-(p*uu).sum(-1,keepdim=True)
        cov=(p*dc*uc).sum(-1,keepdim=True)
        vd=(p*dc.square()).sum(-1,keepdim=True)
        vu=(p*uc.square()).sum(-1,keepdim=True)
    _finite(cov,vd,vu)
    return cov,vd,vu


def _ratio(numerator, denominator, zero=0.):
    safe=torch.where(denominator>0,denominator,torch.ones_like(denominator))
    value=torch.where(denominator>0,numerator/safe,torch.full_like(numerator,zero))
    _finite(value)
    return value


def _frozen_parameters(recipe):
    kind=recipe['kind']
    expected={'final':4,'late':5,'reference':2 if kind=='ref2_late' else 1,'omega':.5}
    if kind in {'ref2_late','fisher_cosine'}: expected['gamma']=1.
    for key,value in expected.items():
        if key in recipe and (isinstance(recipe[key],bool) or recipe[key]!=value):
            raise ValueError('R2 frozen parameter mismatch: '+key)
    if 'gamma' in recipe and kind not in {'ref2_late','fisher_cosine'}:
        raise ValueError('gamma is not defined for this R2 recipe')


@torch.no_grad()
def recipe_logits(logits, recipe):
    kind=recipe['kind']
    if kind in {'fixed','adaptive'}:
        return original.recipe_logits(logits,recipe)
    if kind not in KINDS: raise ValueError('Unknown R2 kind: '+str(kind))
    _frozen_parameters(recipe)
    z=logits[4]; early=logits[2 if kind=='ref2_late' else 1]; late=logits[5]
    for value in (z,early,late):
        original._matrix(value,'logits')
        if value.shape!=z.shape or value.device!=z.device:
            raise ValueError('R2 logits must share target positions/vocabulary/device')
    d=z-early; u=late-z
    _finite(d,u)
    if kind=='ref2_late':
        result=z+.5*d+(z-late)
    else:
        cov,vd,vu=geometry(z,d,u,uniform=kind=='euclidean_signed')
        c=_ratio(cov,vu).clamp(0. if kind=='fisher_positive' else -1.,1.)
        if kind=='fisher_cosine':
            # sqrt separately avoids an unnecessary vd*vu overflow.
            denominator=vd.sqrt()*vu.sqrt(); _finite(denominator)
            strength=_ratio(cov,denominator).clamp(0.,1.)
            result=(z+.5*d)-strength*u
        elif kind=='fisher_matched_norm':
            residual=d-c*u
            _,vr,_=geometry(z,residual,u)
            strength=_ratio(vr,vd,zero=1.).clamp(0.,1.).sqrt()
            result=z+.5*strength*d
        else:
            result=z+.5*(d-c*u)
    _finite(result)
    return result


@torch.no_grad()
def score_logit_recipes(logits, targets, recipes):
    """Full-vocabulary FP32 log_softmax, gather ALL targets, sum; no char norm.

    fixed/adaptive are delegated to the unchanged original scorer, including
    its exact arithmetic. Recipes are sequential to bound vocabulary memory.
    Inputs are never modified and inputs requiring gradients are rejected.
    """
    recipes=list(recipes); ids=[r.get('id') for r in recipes]
    if any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('Unique nonempty recipe IDs required')
    if not recipes:return {}
    target=torch.as_tensor(targets)
    if target.ndim!=1 or target.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
        raise ValueError('Targets must be a 1D integer sequence')
    output={}
    for recipe in recipes:
        if recipe['kind'] in {'fixed','adaptive'}:
            delegated=original.score_logit_recipes(logits,targets,[recipe])
            if not all(math.isfinite(value) for value in delegated.values()):
                raise ValueError('Nonfinite delegated continuation sum')
            output.update(delegated);continue
        z=recipe_logits(logits,recipe)
        if target.numel()!=z.shape[0] or target.numel()==0:
            raise ValueError('One target required for every supplied prediction position')
        t=target.to(device=z.device,dtype=torch.long)
        if int(t.min())<0 or int(t.max())>=z.shape[1]:raise ValueError('Target outside vocabulary')
        lp=z.log_softmax(-1).gather(-1,t[:,None]).squeeze(-1)
        _finite(lp,lp.sum())
        output[recipe['id']]=float(lp.sum())
    return output
