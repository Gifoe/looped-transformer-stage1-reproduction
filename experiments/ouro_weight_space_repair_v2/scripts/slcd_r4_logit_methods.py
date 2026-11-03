"""R4 fixed-budget, training-free scalar guidance; FP32 and original weights.

Fisher RMS bounds the LOCAL quadratic energy, not exact KL/calibration/error.
The KL readout computes actual full-vocabulary KL(p4||p_w), with 16 fixed
bisections on [0,1] and returns the feasible lower endpoint (width <=2**-16).
No gradients, fitted parameters, model loading, candidate labels, or datasets.
"""
import math
import torch
import slcd_logit_methods as original

KINDS={'fisher_rms','kl_trust_region','uniform_rms'}


def _finite(*values):
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError('Nonfinite R4 intermediate or output')


def _pair(final,contrast):
    original._matrix(final,'final');original._matrix(contrast,'contrast')
    if final.shape!=contrast.shape or final.device!=contrast.device:
        raise ValueError('R4 tensors must share shape/device')


@torch.no_grad()
def contrast_variance(final,contrast,*,uniform=False):
    _pair(final,contrast)
    # Exact constant-vector zero variance, independent of probability sum error.
    shifted=contrast-contrast[:,:1]
    if uniform:
        centered=shifted-shifted.mean(-1,keepdim=True)
        variance=centered.square().mean(-1,keepdim=True)
    else:
        p=final.softmax(-1)
        centered=shifted-(p*shifted).sum(-1,keepdim=True)
        variance=(p*centered.square()).sum(-1,keepdim=True)
    _finite(variance)
    return variance


def _kl_from_logp(logp,p,final,contrast,strength):
    proposal=final+strength*contrast;_finite(proposal)
    logq=proposal.log_softmax(-1)
    value=(p*(logp-logq)).sum(-1,keepdim=True)
    _finite(logq,value)
    # FP32 can produce a tiny negative value for a mathematically zero KL.
    return value.clamp_min(0.)


@torch.no_grad()
def kl_at_strength(final,contrast,strength):
    """Actual forward KL in FP32 for the exact logits used in the readout."""
    _pair(final,contrast)
    if (not isinstance(strength,torch.Tensor) or strength.dtype!=torch.float32
        or strength.device!=final.device or strength.shape!=(final.shape[0],1)
        or strength.requires_grad):
        raise ValueError('Strength must be detached FP32 [ntoken,1] on same device')
    _finite(strength)
    logp=final.log_softmax(-1);p=final.softmax(-1);_finite(logp,p)
    return _kl_from_logp(logp,p,final,contrast,strength)


@torch.no_grad()
def guidance_strength(final,contrast,kind):
    """Return fixed-rule [ntoken,1] strengths. No learned/scanned coefficient."""
    _pair(final,contrast)
    if kind not in KINDS:raise ValueError('Unknown R4 guidance kind')
    if kind in {'fisher_rms','uniform_rms'}:
        variance=contrast_variance(final,contrast,uniform=kind=='uniform_rms')
        safe=torch.where(variance>0,variance,torch.ones_like(variance))
        strength=torch.where(variance>0,(.5/safe.sqrt()).clamp_max(1.),
                             torch.full_like(variance,.5))
    else:
        logp=final.log_softmax(-1);p=final.softmax(-1);_finite(logp,p)
        high=torch.ones((final.shape[0],1),dtype=torch.float32,device=final.device)
        cap_kl=_kl_from_logp(logp,p,final,contrast,high)
        cap_feasible=cap_kl<=.125
        low=torch.where(cap_feasible,high,torch.zeros_like(high))
        for _ in range(16):
            middle=(low+high)*.5
            feasible=_kl_from_logp(logp,p,final,contrast,middle)<=.125
            low=torch.where(feasible,middle,low)
            high=torch.where(feasible,high,middle)
        strength=low
        # Recheck the returned actual FP32 logits, not a quadratic surrogate.
        final_kl=_kl_from_logp(logp,p,final,contrast,strength)
        if not (final_kl<=.125).all():raise ValueError('Returned FP32 KL exceeds budget')
    _finite(strength)
    return strength


def _frozen(recipe):
    expected={'final':4,'reference':1,'cap':1.}
    if recipe['kind']=='kl_trust_region':expected.update(kl_budget=.125,bisection_steps=16)
    else:expected.update(rms_budget=.5,zero_variance_strength=.5)
    for key,value in expected.items():
        if key in recipe and (isinstance(recipe[key],bool) or recipe[key]!=value):
            raise ValueError('R4 frozen parameter mismatch: '+key)
    if any(key in recipe for key in ('late','gamma','omega')):
        raise ValueError('No late/gamma/omega parameter in R4 base recipes')


@torch.no_grad()
def recipe_logits(logits,recipe):
    kind=recipe['kind']
    if kind in {'fixed','adaptive'}:return original.recipe_logits(logits,recipe)
    if kind not in KINDS:raise ValueError('Unknown R4 recipe kind')
    _frozen(recipe)
    final=logits[4];reference=logits[1];_pair(final,reference)
    contrast=final-reference;_finite(contrast)
    strength=guidance_strength(final,contrast,kind)
    result=final+strength*contrast;_finite(result)
    return result


@torch.no_grad()
def score_logit_recipes(logits,targets,recipes):
    """Full-vocabulary FP32 log_softmax, gather every target, sum; no char norm.

    Baselines delegate to the unchanged original implementation. Recipes run
    sequentially. Bisection bounds every token separately, not the sequence KL.
    """
    recipes=list(recipes);ids=[r.get('id') for r in recipes]
    if any(not isinstance(i,str) or not i for i in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Unique nonempty recipe IDs required')
    if not recipes:return {}
    target=torch.as_tensor(targets)
    if target.ndim!=1 or target.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
        raise ValueError('Targets must be a 1D integer sequence')
    output={}
    for recipe in recipes:
        if recipe['kind'] in {'fixed','adaptive'}:
            values=original.score_logit_recipes(logits,targets,[recipe])
            if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite baseline sum')
            output.update(values);continue
        z=recipe_logits(logits,recipe)
        if target.numel()!=z.shape[0] or target.numel()==0:
            raise ValueError('One target per supplied prediction position required')
        t=target.to(device=z.device,dtype=torch.long)
        if int(t.min())<0 or int(t.max())>=z.shape[1]:raise ValueError('Target outside vocabulary')
        lp=z.log_softmax(-1).gather(-1,t[:,None]).squeeze(-1)
        total=lp.sum();_finite(lp,total);output[recipe['id']]=float(total)
    return output
