"""CPU-only synthetic tests. No models, datasets, heldout files, or GPU calls.

Default output is stdout; optional --output must name an slcd_ file in an
existing directory. Existing identical evidence is accepted, never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import slcd_logit_methods as original
import slcd_r2_logit_methods as r2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(2021)
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/slcd_R2_RECIPES.json').read_text(encoding='utf-8'))
    recipes=config['recipes']; results={}
    logits={k:torch.randn(4,31,dtype=torch.float32) for k in range(1,7)}
    targets=torch.tensor([0,7,18,30]); before={k:v.clone() for k,v in logits.items()}
    scores=r2.score_logit_recipes(logits,targets,recipes)
    controls=original.score_logit_recipes(logits,targets,recipes[:2])
    assert all(scores[k]==v for k,v in controls.items())
    assert all(torch.equal(logits[k],v) for k,v in before.items())
    assert all(not r2.recipe_logits(logits,r).requires_grad for r in recipes)
    results['baseline_exact_and_no_mutation']=True

    # Constant late update => exactly zero Fisher/Euclidean variance, c=0.
    zero={k:v.clone() for k,v in logits.items()};zero[5]=zero[4].clone()
    fixed=original.recipe_logits(zero,recipes[0])
    for recipe in recipes[2:7]:assert torch.equal(r2.recipe_logits(zero,recipe),fixed)
    z=torch.zeros(2,4);d=torch.tensor([[1.,-1.,2.,-2.],[2.,0.,-2.,0.]])
    u=torch.ones_like(z)*3
    cov,vd,vu=r2.geometry(z,d,u)
    assert torch.equal(vu,torch.zeros_like(vu)) and torch.equal(cov,torch.zeros_like(cov))
    orth={1:z-d,2:z-d,4:z,5:torch.tensor([[1.,1.,-1.,-1.],[0.,2.,0.,-2.]])}
    # In this uniform-p example d and u are orthogonal with nonzero variance.
    cov,_,_=r2.geometry(z,d,orth[5]);assert torch.equal(cov,torch.zeros_like(cov))
    assert torch.equal(r2.recipe_logits(orth,recipes[2]),original.recipe_logits(orth,recipes[0]))
    results['zero_variance_and_zero_coefficient_exact_fixed']=True

    # Nontrivial projection, with deliberately |c|<1 (no clipping).
    u=torch.tensor([[1.,-1.,1.,-1.],[2.,-2.,2.,-2.]])
    v=torch.tensor([[1.,1.,-1.,-1.],[1.,1.,-1.,-1.]])
    d=.25*u+v; cov,vd,vu=r2.geometry(z,d,u);c=cov/vu;g=d-c*u
    residual_cov,vg,_=r2.geometry(z,g,u)
    assert torch.max(residual_cov.abs())<1e-6
    example={1:z-d,2:z-d,4:z,5:u}
    result=r2.recipe_logits(example,recipes[2]);assert torch.allclose(2*(result-z),g,atol=1e-6)
    damp=r2.recipe_logits(example,recipes[5]);_,damp_var,_=r2.geometry(z,2*(damp-z),u)
    assert torch.allclose(damp_var,vg,atol=1e-6,rtol=1e-6)
    results['unclipped_orthogonality_max_abs_cov']=float(residual_cov.abs().max())
    results['matched_fisher_variance_max_error']=float((damp_var-vg).abs().max())
    clipped={1:z-3*u,2:z,4:z,5:u}
    clipped_g=2*(r2.recipe_logits(clipped,recipes[2])-z)
    clip_cov,_,_=r2.geometry(z,clipped_g,u)
    assert (clip_cov>0).all() # Explicit counterexample to exact orthogonality.
    assert torch.equal(clipped_g,2*u)
    results['clipped_is_not_strictly_orthogonal']=True
    negative={1:z+u,2:z,4:z,5:u}
    assert torch.equal(r2.recipe_logits(negative,recipes[3]),z-.5*u)
    assert torch.equal(r2.recipe_logits(negative,recipes[2]),z)
    # Cosine cancellation direction: d=u gives z+.5u-u=z-.5u.
    positive={1:z-u,2:z,4:z,5:u}
    assert torch.equal(r2.recipe_logits(positive,recipes[6]),z-.5*u)
    results['signed_positive_and_cosine_direction']=True

    shifted={k:value+float(k)/4 for k,value in logits.items()}
    shifts=r2.score_logit_recipes(shifted,targets,recipes)
    shift_error=max(abs(scores[k]-shifts[k]) for k in scores)
    assert shift_error<2e-5
    results['score_shift_invariance_max_error']=shift_error
    flat={k:torch.zeros(4,31) for k in range(1,7)}
    flat_scores=r2.score_logit_recipes(flat,targets,recipes)
    assert len(set(flat_scores.values()))==1
    zero_d={k:value.clone() for k,value in logits.items()};zero_d[1]=zero_d[4].clone()
    assert torch.equal(r2.recipe_logits(zero_d,recipes[5]),zero_d[4])
    results['zero_contrast_and_flat_finite']=True

    z1,z2,z4,z5=[logits[k] for k in [1,2,4,5]]
    expected=z4+.5*(z4-z2)+(z4-z5)
    assert torch.equal(r2.recipe_logits(logits,recipes[7]),expected)
    def ll(z):return z.log_softmax(-1).gather(1,targets[:,None]).sum().item()
    ref2=ll(expected)
    a1=ll(z4+.5*(z4-z1))+ll(z4)-ll(z5)
    a2=ll(z4+.5*(z4-z1)+(z4-z5))
    assert abs(ref2-a1)>1e-4 and abs(ref2-a2)>1e-4
    results['ref2_vs_A1_score_difference']=ref2-a1
    results['ref2_vs_A2_ref1_token_difference']=ref2-a2
    assert scores[recipes[7]['id']]==ref2
    results['full_vocabulary_all_targets_exact']=True
    bad={k:v.clone() for k,v in logits.items()};bad[4][0,0]=float('nan')
    rejected=0
    for mutation in ('nan','grad','shape','targets','parameter','duplicate'):
        try:
            if mutation=='nan':r2.score_logit_recipes(bad,targets,recipes)
            elif mutation=='grad':r2.score_logit_recipes({**logits,4:logits[4].clone().requires_grad_()},targets,recipes)
            elif mutation=='shape':r2.score_logit_recipes({**logits,5:logits[5][:2]},targets,recipes)
            elif mutation=='targets':r2.score_logit_recipes(logits,targets[:-1],recipes)
            elif mutation=='parameter':r2.recipe_logits(logits,{**recipes[2],'omega':.75})
            else:r2.score_logit_recipes(logits,targets,[recipes[2],recipes[2]])
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid input accepted: '+mutation)
    results['invalid_inputs_rejected']=rejected
    results['passed']=True;results['device']='cpu';results['data']='synthetic_only'
    results['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(r2.__file__),Path(original.__file__),root/'configs/slcd_R2_RECIPES.json']}
    text=json.dumps(results,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        destination=Path(args.output)
        if not destination.name.startswith('slcd_') or not destination.parent.is_dir():
            raise ValueError('Evidence must use slcd_ filename and an existing parent')
        if destination.exists():
            if destination.read_text(encoding='utf-8')!=text:raise FileExistsError('Preserve existing evidence')
        else:destination.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
