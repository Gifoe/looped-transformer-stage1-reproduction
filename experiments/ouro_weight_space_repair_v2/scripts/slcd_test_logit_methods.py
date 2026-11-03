"""Pure CPU synthetic checks. Never opens datasets, model weights or result logs."""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import torch
from slcd_logit_methods import decompose_contrast,recipe_logits,score_logit_recipes


def main():
    torch.set_num_threads(2);torch.manual_seed(2021)
    logits={k:torch.randn(7,19).bfloat16().float() for k in range(1,7)}
    targets=torch.tensor([0,2,4,6,8,10,18])
    original={k:x.clone() for k,x in logits.items()}
    recipes=[dict(id='fixed',kind='fixed',omega=.5),dict(id='adaptive',kind='adaptive',omega=1.),
             dict(id='two_zero',kind='two_sided_logit',omega=.5,gamma=0),
             dict(id='two_half',kind='two_sided_logit',omega=.5,gamma=.5)]
    for centered in (True,False):
        for kind in ('orthogonal','parallel'):
            recipes.append(dict(id=kind+'_'+str(centered),kind=kind,omega=.5,centered=centered))
    scores=score_logit_recipes(logits,targets,recipes)
    final,reference,late=logits[4],logits[1],logits[5]
    fixed=final+.5*(final-reference)
    direct=float(fixed.log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1).sum())
    assert scores['fixed']==scores['two_zero']==direct
    assert torch.equal(recipe_logits({1:reference,4:final},recipes[2]),fixed)
    assert torch.equal(recipe_logits(logits,recipes[3]),fixed+.5*(final-late))
    # A growing late logit must be subtracted rather than extrapolated upward.
    direction_logits={1:torch.zeros(1,3),4:torch.zeros(1,3),5:torch.tensor([[0.,2.,0.]])}
    assert torch.equal(recipe_logits(direction_logits,recipes[3]),torch.tensor([[0.,-1.,0.]]))
    top=final.softmax(-1).topk(2,dim=-1).values
    adaptive=final+(1-(top[:,0]-top[:,1]))[:,None]*(final-reference)
    assert scores['adaptive']==float(adaptive.log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1).sum())
    p,o=decompose_contrast(final,reference,centered=True)
    a=final-final.mean(-1,keepdim=True);d=final-reference;d=d-d.mean(-1,keepdim=True)
    residual=float((o*a).sum(-1).abs().max())
    assert torch.allclose(p+o,d,atol=1e-6,rtol=1e-6)
    assert residual<1e-4 and float(o.mean(-1).abs().max())<1e-6
    # Independent additive constants per position and depth leave probabilities invariant.
    shifted={k:x+torch.arange(7,dtype=torch.float32)[:,None]+3*k for k,x in logits.items()}
    shift_errors={}
    for recipe in recipes:
        z=recipe_logits(logits,recipe).log_softmax(-1)
        zs=recipe_logits(shifted,recipe).log_softmax(-1)
        error=float((z-zs).abs().max());shift_errors[recipe['id']]=error
        if recipe.get('centered',True):assert error<2e-5,(recipe,error)
    assert shift_errors['orthogonal_False']>1e-3
    zero_p,zero_o=decompose_contrast(torch.full((1,4),3.),torch.tensor([[1.,2.,3.,4.]]))
    assert torch.equal(zero_p,torch.zeros_like(zero_p))
    assert torch.allclose(zero_o,torch.tensor([[1.5,.5,-.5,-1.5]]))
    def lp(x):return float(x.log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1).sum())
    score_level=lp(final)+.5*(lp(final)-lp(reference))+.5*(lp(final)-lp(late))
    assert abs(scores['two_half']-score_level)>1e-3
    # An unrelated different reference/late uses the same declared formula, no search.
    alt=dict(id='synthetic_alternate',kind='two_sided_logit',reference=2,final=4,late=6,omega=.5,gamma=.25)
    assert torch.equal(recipe_logits(logits,alt),final+.5*(final-logits[2])+.25*(final-logits[6]))
    assert all(torch.equal(logits[k],x) for k,x in original.items())
    for invalid in ([dict(id='x',kind='fixed'),dict(id='x',kind='fixed')],
                    [dict(id='x',kind='orthogonal',gamma=.25)]):
        try:score_logit_recipes(logits,targets,invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid recipe accepted')
    root=Path(__file__).resolve().parents[1]
    evidence=dict(status='passed',scope='pure CPU synthetic; no model/dataset/result access',
        torch_version=torch.__version__,seed=2021,full_vocabulary_size=19,target_positions=7,
        gamma0_fixed_exact=True,late_update_subtraction_direction=True,adaptive_cap1_exact=True,
        centered_orthogonality_max_abs_dot=residual,shift_logprob_max_abs_errors=shift_errors,
        two_sided_full_logit_score=scores['two_half'],incorrect_score_level_value=score_level,
        score_level_absolute_disagreement=abs(scores['two_half']-score_level),
        zero_centered_norm_defined=True,alternate_reference_late_formula=True,input_tensors_unchanged=True,
        module_sha256=hashlib.sha256((root/'scripts/slcd_logit_methods.py').read_bytes()).hexdigest())
    output=root/'astra/slcd_SYNTHETIC_TEST.json'
    if output.exists():raise FileExistsError('Preserve existing synthetic evidence: '+str(output))
    output.write_text(json.dumps(evidence,indent=2),encoding='utf8')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
