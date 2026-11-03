"""CPU-only synthetic R4 verification; no datasets/model/GPU access."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import slcd_logit_methods as original
import slcd_r4_logit_methods as r4


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(2021)
    root=Path(__file__).resolve().parents[1]
    recipes=json.loads((root/'configs/slcd_R4_RECIPES.json').read_text(encoding='utf-8'))['recipes']
    logits={k:torch.randn(9,37) for k in range(1,7)}
    target=torch.arange(9)*4;before={k:v.clone() for k,v in logits.items()}
    scores=r4.score_logit_recipes(logits,target,recipes)
    controls=original.score_logit_recipes(logits,target,recipes[:2])
    assert all(scores[k]==v for k,v in controls.items())
    assert all(torch.equal(v,logits[k]) for k,v in before.items())
    assert all(not r4.recipe_logits(logits,r).requires_grad for r in recipes)
    results={'baseline_exact_no_mutation_no_grad':True}

    final=torch.tensor([[1.,0.,-1.,2.],[0.,0.,0.,0.]])
    constant=torch.ones_like(final)*2
    for kind in ['fisher_rms','uniform_rms']:
        assert torch.equal(r4.contrast_variance(final,constant,uniform=kind=='uniform_rms'),torch.zeros(2,1))
        assert torch.equal(r4.guidance_strength(final,constant,kind),torch.full((2,1),.5))
    assert torch.equal(r4.guidance_strength(final,constant,'kl_trust_region'),torch.ones(2,1))
    assert torch.equal(r4.guidance_strength(final,torch.zeros_like(final),'kl_trust_region'),torch.ones(2,1))
    flat={1:torch.zeros(2,4),4:torch.zeros(2,4)}
    flat_scores=r4.score_logit_recipes(flat,[0,3],recipes)
    assert len(set(flat_scores.values()))==1
    results['zero_variance_RMS_half_KL_cap_one']=True

    d=torch.tensor([[2.,-2.,1.,-1.],[3.,-1.,2.,-4.]])
    rms_error=0.
    for kind in ['fisher_rms','uniform_rms']:
        weight=r4.guidance_strength(final,d,kind)
        variance=r4.contrast_variance(final,d,uniform=kind=='uniform_rms')
        assert ((weight>=0)&(weight<=1)).all()
        assert torch.allclose(weight.square()*variance,torch.full_like(weight,.25),atol=1e-7)
        doubled=r4.guidance_strength(final,2*d,kind)
        assert torch.allclose(weight*d,doubled*(2*d),atol=1e-6)
        rms_error=max(rms_error,float((weight.square()*variance-.25).abs().max()))
        tiny=r4.guidance_strength(final,.001*d,kind)
        assert torch.equal(tiny,torch.ones_like(tiny))
    results['RMS_energy_max_error']=rms_error
    results['RMS_scaling_and_cap']=True

    # Actual output KL vs independent float64 reference, and near-maximality.
    z=logits[4];d=z-logits[1]
    weight=r4.guidance_strength(z,d,'kl_trust_region')
    kl=r4.kl_at_strength(z,d,weight)
    assert (kl<=.125).all()
    def kl64(w):
        # Evaluate the exact FP32 proposal in higher precision for QA only.
        proposal=(z+w*d).double()
        lp=z.double().log_softmax(-1);lq=proposal.log_softmax(-1)
        return (lp.exp()*(lp-lq)).sum(-1,keepdim=True)
    verified=kl64(weight)
    assert (verified<=.125+2e-6).all()
    low=torch.zeros_like(weight);high=torch.ones_like(weight)
    for _ in range(32):
        middle=(low+high)*.5;feasible=kl64(middle)<=.125
        low=torch.where(feasible,middle,low);high=torch.where(feasible,high,middle)
    root_error=float((weight-low).abs().max())
    assert root_error<2**-16+2e-6
    results['max_FP32_KL']=float(kl.max())
    results['max_independent_FP64_KL']=float(verified.max())
    results['max_weight_error_vs_independent_root']=root_error
    double_weight=r4.guidance_strength(z,2*d,'kl_trust_region')
    assert torch.allclose(weight*d,double_weight*2*d,atol=1e-4,rtol=1e-4)
    assert torch.equal(r4.guidance_strength(z,.001*d,'kl_trust_region'),torch.ones_like(weight))
    results['KL_scaling_and_cap']=True

    # RMS is not an exact KL bound: rare token is greatly upweighted.
    tail_z=torch.tensor([[-10.,0.]])
    tail_d=torch.tensor([[100.,0.]])
    rms=r4.guidance_strength(tail_z,tail_d,'fisher_rms')
    rms_kl=r4.kl_at_strength(tail_z,tail_d,rms)
    exact=r4.guidance_strength(tail_z,tail_d,'kl_trust_region')
    exact_kl=r4.kl_at_strength(tail_z,tail_d,exact)
    assert rms_kl.item()>.125 and exact_kl.item()<=.125
    results['tail_counterexample_RMS_actual_KL']=float(rms_kl)
    results['tail_exact_KL']=float(exact_kl)

    shifted={k:v+float(k)/4 for k,v in logits.items()}
    shifted_scores=r4.score_logit_recipes(shifted,target,recipes)
    shift_error=max(abs(scores[k]-shifted_scores[k]) for k in scores)
    assert shift_error<1e-4
    results['score_shift_invariance_max_error']=shift_error
    for r in recipes:
        zz=r4.recipe_logits(logits,r)
        ll=zz.log_softmax(-1).gather(1,target[:,None]).sum().item()
        assert scores[r['id']]==ll
    results['full_vocabulary_all_target_sum_exact']=True

    rejected=0
    for case in ['nan','grad','shape','targets','budget','iterations','duplicate']:
        try:
            if case=='nan':
                bad={k:v.clone() for k,v in logits.items()};bad[4][0,0]=float('nan')
                r4.score_logit_recipes(bad,target,recipes)
            elif case=='grad':r4.score_logit_recipes({**logits,4:logits[4].clone().requires_grad_()},target,recipes)
            elif case=='shape':r4.score_logit_recipes({**logits,1:logits[1][:2]},target,recipes)
            elif case=='targets':r4.score_logit_recipes(logits,target[:-1],recipes)
            elif case=='budget':r4.recipe_logits(logits,{**recipes[3],'kl_budget':.25})
            elif case=='iterations':r4.recipe_logits(logits,{**recipes[3],'bisection_steps':17})
            else:r4.score_logit_recipes(logits,target,[recipes[3],recipes[3]])
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid input accepted: '+case)
    results.update(invalid_inputs_rejected=rejected,passed=True,device='cpu',data='synthetic_only')
    results['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(r4.__file__),Path(original.__file__),root/'configs/slcd_R4_RECIPES.json']}
    text=json.dumps(results,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        destination=Path(args.output)
        if not destination.name.startswith('slcd_') or not destination.parent.is_dir():
            raise ValueError('Evidence requires slcd_ name and existing parent directory')
        if destination.exists():
            if destination.read_text(encoding='utf-8')!=text:raise FileExistsError('Preserve previous evidence')
        else:destination.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
