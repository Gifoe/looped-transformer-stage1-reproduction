"""CPU-only synthetic R6 verification. Does not read any dataset or results."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import slcd_r6_scores as r6


def methods(f,k4,k5=None):
    value={'loopcd_k4_fixed':{'normalized_scores':list(f)},'native_k4':{'normalized_scores':list(k4)}}
    if k5 is not None:value['native_k5']={'normalized_scores':list(k5)}
    return value


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output');args=ap.parse_args()
    original=methods([-2.,-3.,-1.,-4.],[-3.,-3.5,-2.,-3.],[-3.5,-3.,-2.5,-3.25])
    prior=methods([-1.,-2.,-3.,-4.],[-2.,-2.5,-2.,-4.5])
    before=copy.deepcopy((original,prior));out=r6.all_scores(original,prior)
    f=np.asarray(original['loopcd_k4_fixed']['normalized_scores']);k4=np.asarray(original['native_k4']['normalized_scores']);k5=np.asarray(original['native_k5']['normalized_scores'])
    e=np.asarray(prior['loopcd_k4_fixed']['normalized_scores'])-np.asarray(prior['native_k4']['normalized_scores']);g=f-k4;a=f+(k4-k5)
    assert np.array_equal(out['r6_prior_quarter'],a-.25*e)
    assert np.array_equal(out['r6_prior_half'],a-.5*e)
    assert (original,prior)==before
    assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in out.values())
    results={'quarter_half_exact_formulas':True,'FP64_finite_no_mutation':True}
    for mid in r6.METHOD_IDS:assert np.array_equal(out[mid],r6.selected_score(original,prior,mid))
    for gg,ee,expected in [([1.,-1.],[-1.,1.],0.),([3.,-3.],[1.,-1.],1.),([.25,-.25],[1.,-1.],.25),([1.,-1.],[2.,2.],0.)]:
        alpha,_,_,_=r6.projection_parameters(gg,ee);assert alpha==expected
    alpha,rho,_,_=r6.projection_parameters([4.,4.,4.],[1.,2.,3.]);assert alpha==0. and rho==1.
    alpha,rho,gc,ec=r6.projection_parameters([1.25,.75,-.75,-1.25],[1.,-1.,1.,-1.])
    assert 0<alpha<1
    norm_error=abs(np.linalg.norm(gc-alpha*ec)-np.linalg.norm(rho*gc))
    assert norm_error<1e-12 and abs(np.dot(gc-alpha*ec,ec))<1e-12
    results['alpha_clipping_and_degenerate']=True;results['matched_norm_error']=float(norm_error)
    zero_prior=methods([1.,1.,1.,1.],[0.,0.,0.,0.])
    for mid in ['r6_prior_projection','r6_matched_magnitude']:
        assert np.array_equal(r6.selected_score(original,zero_prior,mid),a)
    results['alpha_zero_exact_A1']=True
    # Independent constant shifts in each source score vector only shift output.
    shifted_o=copy.deepcopy(original);shifted_p=copy.deepcopy(prior)
    for i,v in enumerate(shifted_o.values()):v['normalized_scores']=[x+(i+1)*2 for x in v['normalized_scores']]
    for i,v in enumerate(shifted_p.values()):v['normalized_scores']=[x+(i+1)*4 for x in v['normalized_scores']]
    for mid,value in out.items():
        shifted=r6.selected_score(shifted_o,shifted_p,mid)
        assert np.allclose(shifted-shifted.mean(),value-value.mean(),atol=1e-12,rtol=0)
        assert np.array_equal(np.argsort(shifted),np.argsort(value))
    results['constant_shift_rank_invariance']=True
    original['gold']=3;prior['gold']=0
    for source in (original,prior):
        for value in source.values():
            if isinstance(value,dict):value.update(gold=99,correct=False,prediction=100)
    assert all(np.array_equal(out[mid],r6.selected_score(original,prior,mid)) for mid in out)
    results['gold_and_prediction_metadata_ignored']=True
    rejected=0
    for kind in ['missing','nan','shape','bool','complex','string','unknown','overflow']:
        bad=copy.deepcopy(original)
        if kind=='missing':del bad['native_k4']
        elif kind=='nan':bad['native_k4']['normalized_scores'][0]=float('nan')
        elif kind=='shape':bad['native_k4']['normalized_scores']=[1.,2.]
        elif kind=='bool':bad['native_k4']['normalized_scores']=[True]*4
        elif kind=='complex':bad['native_k4']['normalized_scores']=[1j]*4
        elif kind=='string':bad['native_k4']['normalized_scores']=['1']*4
        elif kind=='overflow':bad['native_k4']['normalized_scores']=[1e308,-1e308,1e308,-1e308]
        try:r6.selected_score(bad,prior,'unknown' if kind=='unknown' else 'r6_prior_projection')
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid input accepted: '+kind)
    results.update(invalid_inputs_rejected=rejected,passed=True,device='cpu',data='synthetic_only')
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/slcd_R6_METHODS.json').read_text(encoding='utf-8'))
    assert tuple(config['methods'])==r6.METHOD_IDS
    results['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(r6.__file__),root/'configs/slcd_R6_METHODS.json']}
    text=json.dumps(results,indent=2)+'\n'
    if args.output:
        p=Path(args.output)
        if not p.name.startswith('slcd_') or not p.parent.is_dir():raise ValueError('Existing parent and slcd_ name required')
        if p.exists():
            if p.read_text(encoding='utf-8')!=text:raise FileExistsError('Preserve existing evidence')
        else:p.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
