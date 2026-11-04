"""Synthetic-only LP and certificate checks. No research data access."""
import json,copy
from types import SimpleNamespace
import numpy as np
from r21_lp import solve_label,certify

def main():
    winner=np.array([[2.,3.],[0.,0.],[-1.,-1.]])
    assert solve_label(winner,0)['status']=='strict_reachable'
    convex=np.array([[0.,0.],[1.,-2.],[-2.,1.]])
    assert all(int(np.argmax(convex[:,k]))!=0 for k in range(2)) and solve_label(convex,0)['status']=='strict_reachable'
    unreachable=np.array([[0.,0.],[1.,2.]])
    assert solve_label(unreachable,0)['status']=='strict_unreachable'
    assert solve_label(np.zeros((3,2)),0)['status']=='uncertain'
    p=[2,0,1];assert solve_label(convex[p],1)['status']==solve_label(convex,0)['status']
    assert solve_label(convex+np.array([17.,-9.]),0)['status']=='strict_reachable'
    assert solve_label(convex*3,0)['status']=='strict_reachable'
    # Known sign: lambda=-ineqlin.marginals must yield positive simplex.
    solved=solve_label(unreachable,0);assert min(solved['raw_dual_weights'])>=0 and solved['U_upper']<0
    A=np.array([[1.,-1.],[-1.,1.]])
    bad=SimpleNamespace(x=np.array([1.,0.,1.]),ineqlin=SimpleNamespace(marginals=np.array([-.5,-.5])))
    assert not certify(A,bad)['certificate_valid']
    bad=SimpleNamespace(x=np.array([.5,.5,0.]),ineqlin=SimpleNamespace(marginals=np.array([.5,.5])))
    try:certify(A,bad)
    except ValueError:pass
    else:raise AssertionError('Wrong dual sign accepted')
    # H7 scale may exceed H6: inclusion certificates must use original units.
    h6=solve_label(convex,0);h7=solve_label(np.column_stack([convex,np.array([0.,100.,100.])]),0)
    assert h7['scale']>h6['scale'] and h7['original_margin_upper']>=h6['original_margin_lower']
    return dict(different_scale_original_units=True,status='passed_CPU_synthetic',existing_winner=True,convex_only=True,unreachable=True,tie=True,permutation=True,column_shift=True,positive_scaling=True,dual_sign=True,bad_certificate_rejected=True,scope='Synthetic only; no real576 oracle, model or GPU')
if __name__=='__main__':print(json.dumps(main()))
