"""Direct joint-option auxiliary view; same original R6 primary readout.

This tests known multiview evidence fusion, not a novel fusion principle.
Each control is norm matched to its corresponding protected increment.
"""
import numpy as np
from r8_scores import center,norm,finite,pair_margin_score,EPS
from r9_scores import evidence

ELIGIBLE=('joint4_fusion','joint4_protected','jointFixed_protected','balanced4_protected')


def score(a,r,fixed,joint4,jointFixed,balanced4):
    a,r,fixed=[np.asarray(v,dtype=np.float64) for v in [a,r,fixed]]
    assert a.shape==r.shape==fixed.shape;finite(a,r,fixed)
    output={};diagnostics={}
    for name,aux in [('joint4',joint4),('jointFixed',jointFixed),('balanced4',balanced4)]:
        u,ed=evidence(r,aux);protected,pd=pair_margin_score(a,r,r+u,fixed)
        permutation=np.random.default_rng(2021).permutation(len(r))
        control,cd=pair_margin_score(a,r,r+u[permutation],fixed)
        target=norm(center(protected-r));cn=norm(center(control-r));resolved=cn>EPS*max(norm(u),1.)
        control=r+(np.zeros_like(r) if not resolved else target/cn*center(control-r))
        output.update({name+'_protected':protected,name+'_shuffled_control':control})
        if name=='joint4':output[name+'_fusion']=r+u
        diagnostics[name]=dict(evidence=ed,projection=pd,control_projection=cd,control_permutation=permutation.tolist(),
            control_norm_match_possible=bool(resolved or target==0.),control_target_norm=target,
            control_norm=norm(center(control-r)),control_norm_absolute_error=abs(norm(center(control-r))-target))
    finite(*output.values())
    return output,diagnostics
