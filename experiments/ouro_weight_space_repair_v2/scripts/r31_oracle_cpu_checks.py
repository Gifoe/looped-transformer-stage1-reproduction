"""Counterexamples and rational certificate checks, synthetic input only."""
from fractions import Fraction as Q
import json,hashlib,itertools,platform,time
from pathlib import Path
import numpy as np
from r31_oracle_math import exact,TAU,reduction_domain,rounding_bound,coefficients,margins,witness_certificate,normalize_dual,upper_certificate,unpack
from r31_oracle_solver import propose
ROOT=Path(__file__).resolve().parents[1]
def reject(f):
 try:f()
 except AssertionError:return
 raise AssertionError('Negative certificate fixture accepted')
def solve(a,d,b=None):
 a=[exact(x) for x in a];d=[[exact(x) for x in row] for row in d];b=[exact(0)]*len(a) if b is None else [exact(x) for x in b];r=propose(a,d,b)
 if r['witness'] is not None:
  low,l=witness_certificate(a,d,b,r['witness']);assert l==unpack(r['witness_margin_lower']) and low==[unpack(v) for v in r['witness_pair_lowers']]
 if r['dual_weights'] is not None:assert upper_certificate(a,d,b,[unpack(v) for v in r['dual_weights']])==unpack(r['upper'])
 # Exhaustive small binary domain is a separate combinatorial check.
 best=max(min(x+sum((u*v for u,v in zip(row,bits)),Q(0)) for x,row in zip(a,d)) for bits in itertools.product([0,1],repeat=len(d[0])))
 if r['classification']=='certified_reachable':assert best>TAU
 if r['classification']=='certified_unreachable_strict_margin':assert best<=TAU
 return r,best
def main():
 started=time.monotonic();checks=[];records={}
 coupled,best=solve([Q(-3,4),Q(1,4)],[[1],[-1]]);uncoupled,_=solve([Q(-3,4),Q(1,4)],[[1,0],[0,-1]]);assert coupled['classification']=='certified_unreachable_strict_margin' and uncoupled['classification']=='certified_reachable';checks.append('cross_candidate_shared_prefix_coupling_cheat_exposed');records['coupled']=coupled
 mixed,best=solve([Q(-1,2)],[[1,-1]]);assert mixed['classification']=='certified_reachable' and mixed['witness']==[1,0] and Q(-1,2)<=TAU;checks.append('only_mixed_binary_reachable');records['mixed']=mixed
 no,_=solve([-1],[[0]]);assert no['classification']=='certified_unreachable_strict_margin';checks.append('all_choices_unreachable')
 gap,best=solve([Q(-1,4),Q(3,4)],[[1],[-1]]);assert gap['classification']=='undetermined' and best<0 and unpack(gap['upper'])>TAU;checks.append('LP_relaxation_feasible_integer_infeasible_is_undetermined');records['integer_gap']=gap
 for value in [Q(0),TAU,TAU/2]:
  tie,_=solve([value],[[]]);assert tie['classification']=='certified_unreachable_strict_margin'
 yes,_=solve([2*TAU],[[]]);assert yes['classification']=='certified_reachable';checks.append('strict_tau_ties_not_argmax_success')
 reject(lambda:normalize_dual([-.1,1.]));reject(lambda:normalize_dual([0.,0.]));reject(lambda:upper_certificate([Q(0)],[[Q(1)]],[Q(0)],[Q(2)]));w=normalize_dual([-1e-15,1.]);assert w==[Q(0),Q(1)];checks.append('dual_sign_zero_mass_non_simplex_rejected_exact_normalization')
 pieces=[([-2.],[-.5]),([-1.],[-3.]),([-4.],[-4.])];keys=[['shared'],['shared'],['zero']];problem=coefficients(pieces,keys,[1,1,1],[0.,0.,0.]);wrong,a,d,b=margins(problem,0);r=propose(a,d,b);assert r['classification']=='certified_reachable' and problem['zero_keys']==['zero'];checks.append('zero_columns_only_and_endpoints')
 for c,(f,p) in enumerate(pieces):assert problem['a'][c]==sum(map(exact,f),Q(0)) and problem['a'][c]+sum(problem['d'][c],Q(0))==sum(map(exact,p),Q(0))
 perm=[2,0,1];other=coefficients([pieces[i] for i in perm],[keys[i] for i in perm],[1]*3,[0.]*3);_,a,d,b=margins(other,1);assert propose(a,d,b)['classification']==r['classification'];checks.append('candidate_permutation_invariant')
 repeated=coefficients([([-1.],[-.5]),([-1.],[-.5])],[['same'],['same']],[1,1],[0.,0.]);_,a,d,b=margins(repeated,0);assert propose(a,d,b)['classification']=='certified_unreachable_strict_margin';checks.append('duplicate_candidate_cannot_gain_strict_margin')
 max_ratio=Q(0)
 for n in [1,2,3,17,128]:
  f=np.array([-100000.]+[-.01]*(n-1),dtype=np.float32);p=np.array([-99999.]+[-.02]*(n-1),dtype=np.float32);bound=rounding_bound(f,p,7,.125)
  for bits in [[0]*n,[1]*n,[i%2 for i in range(n)]]:
   selected=np.where(bits,p,f).astype('float32');actual=float(selected.sum(dtype=np.float32))/7+.125;ideal=sum((exact(v) for v in selected),Q(0))/7+exact(.125);error=abs(exact(actual)-ideal);assert error<=bound;max_ratio=max(max_ratio,error/bound)
 checks.append('FP32_numpy_reduction_is_only_a_bound_fixture_not_GPU_exact')
 reject(lambda:reduction_domain(2,Q((2**24-1)*2**104)));reject(lambda:reduction_domain(2**23,Q(0)));checks.append('near_max_overflow_and_overlong_reduction_domain_rejected')
 reject(lambda:witness_certificate([Q(0)],[[Q(1)]],[Q(0)],[.999999999999]));checks.append('near_integer_not_silently_rounded')
 result=dict(status='passed_CPU_synthetic_only',checks=checks,records=records,rounding_max_error_bound_ratio=str(max_ratio),elapsed_seconds=time.monotonic()-started,platform=platform.platform(),numpy=np.__version__,GPU_worker_seconds=0,real24_oracle_run=False,model_loaded=False,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/r31_oracle_math.py','scripts/r31_oracle_solver.py','scripts/r31_oracle_cpu_checks.py']})
 with (ROOT/'results/r31_ORACLE_CPU_SYNTHETIC_CHECKS.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main()
