"""Numerical identities and degeneracy checks, no empirical outcomes accessed."""
import json
import numpy as np
from r7_core import ROOT
from r11_scores import repair,all_scores
from r8_scores import center,norm
from slcd_r6_scores import projection_parameters


def main():
    rng=np.random.default_rng(2021);maximum=0.;cases=0
    for n in [4,5]:
        for _ in range(100):
            a,g,e=rng.normal(size=(3,n));alpha,_,_,_=projection_parameters(g,e);r=a-alpha*e;s=rng.normal(size=(4,n));s=np.asarray([center(v) for v in s])
            out,d=repair(a,g,e,r,s,max(norm(v) for v in s));p=rng.permutation(n)
            reordered,_=repair(a[p],g[p],e[p],r[p],s[:,p],max(norm(v) for v in s));err=float(np.max(np.abs(out[p]-reordered)));maximum=max(maximum,err);assert err<1e-12
            zero,_=repair(a,g,e,r,np.zeros((4,n)),1.);assert np.array_equal(zero,r)
            one,d1=repair(a,g,e,r,s[:1],norm(s[0]));duplicate,_=repair(a,g,e,r,np.repeat(s[:1],4,axis=0),norm(s[0]));assert np.max(np.abs(one-duplicate))<1e-12
            assert abs(d1['residual_dot_mean_axis'])<1e-12
            collinear,cd=repair(a,g,e,r,np.asarray([center(e)]),norm(center(e)));assert cd['prior_residual_zero'] and np.array_equal(collinear,a)
            ng,_=repair(a,np.zeros(n),e,a,s,max(norm(v) for v in s));assert np.array_equal(ng,a)
            shifted,_=repair(a+5,g-2,e+3,r+5-alpha*3,s,max(norm(v) for v in s))
            # e's common offset changes only the output's common offset; ranks
            # and centered score changes must be invariant.
            assert np.max(np.abs(center(shifted)-center(out)))<1e-12
            cases+=1
    try:repair(np.ones(4),np.ones(4),np.ones(4),np.ones(4),np.array([[np.nan]*4]),1.)
    except ValueError:pass
    else:raise AssertionError('Nonfinite accepted')
    a=np.arange(4,dtype=float);e=center([1.,0.,-1.,0.]);g=np.array([.5,.1,-.5,-.1]);alpha,_,_,_=projection_parameters(g,e);r=a-alpha*e
    close=e+1e-8*center([0.,1.,0.,-1.]);_,d=repair(a,g,e,r,np.asarray([close]),norm(close))
    assert not d['prior_residual_zero'],'A representable small residual must not be truncated as a rounding zero'
    result=dict(status='passed',synthetic_cases=cases,max_permutation_error=maximum,zero_axis_exact_R6=True,
        zero_residual_exact_A1=True,zero_guidance_exact_A1=True,duplicate_reference_equivalent=True,
        single_reference_residual_orthogonal=True,robust_residual_not_claimed_orthogonal=True,nonfinite_rejected=True,
        representable_near_collinear_residual_preserved=True)
    (ROOT/'results/r11_MATH_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(result,flush=True)


if __name__=='__main__':main()
