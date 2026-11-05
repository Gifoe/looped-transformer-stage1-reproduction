"""Single-thread SciPy proposals; all accepted classifications use Fractions."""
import os,time,math,warnings
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
from r31_oracle_math import TAU,pack,normalize_dual,witness_certificate,upper_certificate
def raw_result(x):
 if isinstance(x,dict):return {str(k):raw_result(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [raw_result(v) for v in x]
 if hasattr(x,'tolist'):return raw_result(x.tolist())
 if isinstance(x,float) and not math.isfinite(x):return repr(x)
 return x
def propose(A,D,B,limit=3.):
 import numpy as np,scipy,scipy.optimize as opt
 from scipy.optimize._highspy import _core
 highs_version='.'.join(str(getattr(_core,'HIGHS_VERSION_'+k)) for k in ['MAJOR','MINOR','PATCH'])
 assert 0<limit<=3 and len(A)==len(D)==len(B)>0;n=len(D[0]);matrix=np.column_stack([-np.asarray([[float(x) for x in row] for row in D],dtype=float),np.ones(len(A))]);objective=np.zeros(n+1);objective[-1]=-1;lo=np.r_[np.zeros(n),-np.inf];hi=np.r_[np.ones(n),np.inf];messages=[];start=time.monotonic()
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always');mip=opt.milp(objective,integrality=np.r_[np.ones(n),0],bounds=opt.Bounds(lo,hi),constraints=opt.LinearConstraint(matrix,-np.inf,np.asarray([float(x) for x in A])),options=dict(time_limit=float(limit),mip_rel_gap=0.,threads=1,random_seed=2021));messages.extend(str(w.message) for w in ws)
 bits=None;lowers=None;lower=None
 if mip.x is not None and np.isfinite(mip.x).all() and all(x==0. or x==1. for x in mip.x[:n]):
  bits=[int(x) for x in mip.x[:n]];lowers,lower=witness_certificate(A,D,B,bits)
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always');lp=opt.linprog(objective,A_ub=matrix,b_ub=np.asarray([float(a+b) for a,b in zip(A,B)]),bounds=[(0.,1.)]*n+[(None,None)],method='highs',options=dict(time_limit=float(limit),threads=1,random_seed=2021));messages.extend(str(w.message) for w in ws)
 raw=None;weights=None;upper=None;dual_error=None
 if getattr(lp,'ineqlin',None) is not None and lp.ineqlin.marginals is not None:
  raw=[-float(v) for v in lp.ineqlin.marginals]
  try:weights=normalize_dual(raw);upper=upper_certificate(A,D,B,weights)
  except AssertionError as e:dual_error=repr(e)
 reach=lower is not None and lower>TAU;unreach=upper is not None and upper<=TAU;assert not(reach and unreach)
 status='certified_reachable' if reach else 'certified_unreachable_strict_margin' if unreach else 'undetermined'
 def finite_vector(x):return None if x is None else [float(v) if math.isfinite(float(v)) else None for v in x]
 return dict(classification=status,witness=bits,witness_margin_lower=pack(lower) if lower is not None else None,witness_pair_lowers=[pack(x) for x in lowers] if lowers is not None else None,dual_weights=[pack(x) for x in weights] if weights is not None else None,upper=pack(upper) if upper is not None else None,raw_dual=finite_vector(raw),dual_validation_error=dual_error,MILP=dict(status=int(mip.status),message=str(mip.message),x=finite_vector(mip.x)),LP=dict(status=int(lp.status),message=str(lp.message),x=finite_vector(lp.x)),raw_MILP_result=raw_result(dict(mip)),raw_LP_result=raw_result(dict(lp)),solver_warnings=messages,solver_elapsed_seconds=time.monotonic()-start,scipy=scipy.__version__,numpy=np.__version__,HiGHS_version=highs_version,thread_limit=1,MILP_time_limit_seconds=float(limit),scope='Witness/dual proposal only; solver status is never an infeasibility or reachability proof')
