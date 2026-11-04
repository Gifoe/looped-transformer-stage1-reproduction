"""Synthetic structural checks before development feedback; no model/data."""
import numpy as np
from r8_scores import METHODS,selected_score,center,norm,unit
from r8_core import sha,save


def main():
    rng=np.random.default_rng(2021);checks=[];max_protected=0.;max_match=0.
    for _ in range(100):
        n=4;a=rng.normal(size=n);r=a+rng.normal(size=n);m=r+rng.normal(size=n);ell=rng.normal(size=n);h=rng.normal(size=n)
        values={name:selected_score(a,r,m,ell,h,name) for name in METHODS};c=unit(center(r-a))
        for name,(score,d) in values.items():
            assert np.isfinite(score).all() and np.isclose(score.mean(),r.mean(),atol=2e-14)
            shifted,_=selected_score(a+8,r+8,m+8,ell,h,name);assert np.allclose(shifted,score+8,atol=2e-13)
            p=rng.permutation(n);permuted,_=selected_score(a[p],r[p],m[p],ell[p],h[p],name);assert np.allclose(permuted,score[p],atol=2e-13)
        for name in ['keep_r6_axis','late_orthogonal','matched_late']:
            err=abs(center(values[name][0]-r)@c);max_protected=max(max_protected,err);assert err<=2e-13
        assert center(values['nonundo_r6_axis'][0]-r)@c>=-2e-13
        for original,control in [('keep_r6_axis','matched_keep'),('nonundo_r6_axis','matched_nonundo')]:
            err=abs(norm(center(values[original][0]-r))-norm(center(values[control][0]-r)));max_match=max(max_match,err);assert err<=2e-13
        for name in ['keep_r6_axis','nonundo_r6_axis']:assert norm(center(values[name][0]-r))<=norm(center(m-r))+2e-13
    checks+=['full axis preservation','nonundo halfspace','increment norms do not expand','matched increment norms',
             'score gauge shift','choice permutation','finite outputs','mean score preserved']
    a=np.zeros(4);r=np.array([1.,-1.,0.,0.]);m=r.copy();ell=np.zeros(4);h=np.zeros(4)
    for name in METHODS:assert np.array_equal(selected_score(a,r,m,ell,h,name)[0],r)
    checks.append('zero signal exact R6')
    ell=np.array([2.,-2.,0.,0.]);h=ell.copy()
    assert np.array_equal(selected_score(a,r,m,ell,h,'late_orthogonal')[0],r)
    checks.append('collinear protected late axis resolves to zero')
    assert len(__import__('r8_scores').partitions(4))==15 and len(__import__('r8_scores').partitions(5))==52
    a=np.array([3.,2.,1.,0.]);r=np.array([2.,3.,1.,0.]);m=np.array([4.,2.,1.,0.]);f=a.copy()
    s,d=selected_score(a,r,m,np.zeros(4),None,'preserve_pair_margins',fixed=f)
    assert all(s[i]-s[j]>=r[i]-r[j]-1e-14 for i,j in d['edges'])
    assert d['protected_pair_count']>0 and s.argmax()==r.argmax()
    same,d=selected_score(r,r,m,np.zeros(4),None,'preserve_pair_margins',fixed=r)
    assert same.argmax()==m.argmax() and d['protected_pair_count']==0 and d['removed_increment_norm']==0.
    checks+=['all set partitions enumerated','changed pair margins preserved','no edge keeps M2 ranking']
    try:selected_score(a,r,np.full(4,np.nan),ell,h,'keep_r6_axis')
    except ValueError:checks.append('nonfinite errors are rejected')
    else:raise AssertionError('Nonfinite input accepted')
    save('results/r8_MATH_CHECKS.json',dict(status='passed',checks=checks,random_cases=100,max_axis_error=max_protected,max_matched_norm_error=max_match,
        scope='CPU score geometry only, not empirical accuracy or token-logit reconstruction',source_sha256={n:sha('scripts/'+n) for n in ['r8_scores.py','r8_checks.py']}))
    print('R8_MATH_CHECKS_PASSED',max_protected,max_match,flush=True)


if __name__=='__main__':main()
