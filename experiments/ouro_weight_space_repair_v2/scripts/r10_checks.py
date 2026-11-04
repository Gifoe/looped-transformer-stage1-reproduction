"""Pre-pilot numerical invariants and cyclic-position mapping identity."""
import numpy as np
from r10_core import save,joint_requests
from r10_scores import score,ELIGIBLE


def main():
    rng=np.random.default_rng(2021);maxerr=0.;unmatched={k:0 for k in ['joint4','jointFixed','balanced4']}
    for n in [4,5]:
        for _ in range(100):
            a,r,f,b,bf,bb=rng.normal(size=(6,n));s,d=score(a,r,f,b,bf,bb);p=rng.permutation(n)
            sp,_=score(a[p],r[p],f[p],b[p],bf[p],bb[p])
            for k in ELIGIBLE:
                e=float(np.max(np.abs(sp[k]-s[k][p])));maxerr=max(maxerr,e);assert e<1e-12
            shifted,_=score(a+3,r+3,f+3,b+4,bf-2,bb+7)
            for k in s:assert np.max(np.abs(shifted[k]-3-s[k]))<1e-12
            for k in d:
                if d[k]['control_norm_match_possible']:assert d[k]['control_norm_absolute_error']<1e-12
                else:unmatched[k]+=1
                for i,j in d[k]['projection']['edges']:assert s[k+'_protected'][i]-s[k+'_protected'][j]>=r[i]-r[j]-1e-12
            position_bias=rng.normal(size=n);content=rng.normal(size=n);mapped=[]
            for shift in range(n):
                order=[(i+shift)%n for i in range(n)];observed=content[order]+position_bias;v=np.empty(n)
                for j,i in enumerate(order):v[i]=observed[j]
                mapped.append(v)
            assert np.max(np.abs(np.mean(mapped,axis=0)-content-position_bias.mean()))<1e-12
    clean=dict(question='Question text?',choices=['one','two','three','four'])
    assert joint_requests(clean,'prefix')==joint_requests(dict(clean,gold=2,correct=False),'prefix')
    save('results/r10_MATH_CHECKS.json',dict(status='passed',random_cases=200,max_permutation_error=maxerr,
        synthetic_cyclic_position_mapping_exact=True,common_shift_invariant=True,gold_metadata_ignored=True,
        control_unmatched_cases=unmatched))
    print('R10_MATH_CHECKS_PASSED',maxerr,flush=True)


if __name__=='__main__':main()
