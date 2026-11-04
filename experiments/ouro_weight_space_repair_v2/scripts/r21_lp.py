"""R21 CPU LP certificates; oracle only, never a deployable coefficient rule."""
import numpy as np
TAU=1e-8

def certify(A,result):
    A=np.asarray(A,dtype=np.float64);m,d=A.shape;raw_w=np.asarray(result.x[:-1],dtype=np.float64);raw_t=float(result.x[-1]);raw_l=-np.asarray(result.ineqlin.marginals,dtype=np.float64)
    if not all(np.isfinite(x).all() for x in [A,raw_w,raw_l,np.array(raw_t)]):raise ValueError('Nonfinite LP certificate')
    wp=np.maximum(raw_w,0);lp=np.maximum(raw_l,0)
    if wp.sum()<=0 or lp.sum()<=0:raise ValueError('Nonpositive simplex denominator')
    w=wp/wp.sum();lam=lp/lp.sum();L=float(np.min(A@w));U=float(np.max(A.T@lam));delta=float(64*np.finfo(np.float64).eps*(d+m+2)*max(1,float(np.max(np.abs(A)))))
    residual=dict(primal_sum=abs(float(raw_w.sum())-1),primal_negative=max(0.,float(-raw_w.min())),primal_inequality=max(0.,raw_t-float(np.min(A@raw_w))),dual_sum=abs(float(raw_l.sum())-1),dual_negative=max(0.,float(-raw_l.min())),primal_objective_mismatch=abs(raw_t-L),dual_gap=U-L)
    valid=max(abs(x) for x in residual.values())<=TAU and U+delta>=L-delta
    lower=L-delta;upper=U+delta;status='strict_reachable' if valid and lower>TAU else 'strict_unreachable' if valid and upper<-TAU else 'uncertain'
    return dict(status=status,primal_weights=w.tolist(),dual_weights=lam.tolist(),raw_primal_weights=raw_w.tolist(),raw_dual_weights=raw_l.tolist(),raw_solver_margin=raw_t,raw_residuals=residual,L=L,U=U,roundoff_delta=delta,L_lower=lower,U_upper=upper,certificate_valid=bool(valid))

def solve_label(scores,label):
    from scipy.optimize import linprog
    scores=np.asarray(scores,dtype=np.float64);n,d=scores.shape;assert n>=2 and d>=1 and np.isfinite(scores).all()
    shifted=scores-scores[:1];scale=max(1.,float(np.max(scores.max(axis=0)-scores.min(axis=0))));assert scale>0
    normalized=shifted/scale;A=np.stack([normalized[label]-normalized[j] for j in range(n) if j!=label]);c=np.r_[np.zeros(d),-1.]
    result=linprog(c,A_ub=np.column_stack([-A,np.ones(n-1)]),b_ub=np.zeros(n-1),A_eq=np.array([np.r_[np.ones(d),0.]]),b_eq=[1.],bounds=[(0,None)]*d+[(None,None)],method='highs')
    if not result.success:raise RuntimeError('Solver failure: '+str(result.message))
    cert=certify(A,result);cert.update(label=int(label),scale=scale,original_margin_lower=cert['L_lower']*scale,original_margin_upper=cert['U_upper']*scale,solver_message=result.message)
    return cert
