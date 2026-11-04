"""Meaningful score and data-isolation invariants before actual pilot."""
import numpy as np
from r9_scores import score
from r9_core import save,reverse_requests


def main():
    rng=np.random.default_rng(2021);maxerror=0.;matched=0;unmatched=0
    for n in [4,5]:
        for _ in range(100):
            a,r,f,b,bf=[rng.normal(size=n) for _ in range(5)]
            s,d=score(a,r,f,b,bf);p=rng.permutation(n)
            ss,dd=score(a[p],r[p],f[p],b[p],bf[p])
            for k in ['reverse4_fusion','reverse4_protected','reverseFixed_protected']:
                error=float(np.max(np.abs(ss[k]-s[k][p])));maxerror=max(maxerror,error);assert error<1e-12
            shifted,_=score(a+3,r+3,f+3,b+5,bf-2)
            for k in s:assert np.max(np.abs(shifted[k]-3-s[k]))<1e-12
            if d['control_norm_match_possible']:
                matched+=1;assert d['control_norm_absolute_error']<1e-12
            else:unmatched+=1
            for k in ['reverse4_protected','reverseFixed_protected']:
                for i,j in d['protected']['edges']:assert s[k][i]-s[k][j]>=r[i]-r[j]-1e-12
    a,r,f=rng.normal(size=(3,4));zero,_=score(a,r,f,np.ones(4),np.zeros(4))
    for v in zero.values():assert np.array_equal(v,r)
    clean=dict(question='Question text?',choices=['one','two','three','four'])
    assert reverse_requests(clean,'prefix')==reverse_requests(dict(clean,gold=1,correct=True),'prefix')
    try:score(a,r,f,np.array([np.nan,0.,0.,0.]),np.zeros(4))
    except ValueError:pass
    else:raise AssertionError('Nonfinite accepted')
    save('results/r9_MATH_CHECKS.json',dict(status='passed',random_cases=200,max_permutation_error=maxerror,
        control_matched_cases=matched,control_unmatched_cases=unmatched,constant_auxiliary_recovers_R6=True,
        common_shift_invariant=True,nonfinite_rejected=True,gold_metadata_ignored=True))
    print('R9_MATH_CHECKS_PASSED',maxerror,matched,unmatched,flush=True)


if __name__=='__main__':main()
