"""Exact rational certificates for shared-prefix binary capacity; no model."""
from fractions import Fraction as Q
import math
TAU=Q(1,100000000)
NEGATIVE_DUAL_CLIP=1e-12
def exact(x):
 if isinstance(x,Q):return x
 if isinstance(x,int):return Q(x)
 v=float(x);assert math.isfinite(v);return Q.from_float(v)
def pack(x):
 x=exact(x);return dict(numerator=str(x.numerator),denominator=str(x.denominator))
def unpack(x):return Q(int(x['numerator']),int(x['denominator']))
def reduction_domain(n,S):
 assert type(n)==int and n>0 and n*Q(1,2**24)<Q(1,2)
 u=Q(1,2**24);gamma=(n-1)*u/(1-(n-1)*u);assert S>=0 and S+gamma*S+n*Q(1,2**149)<=Q((2**24-1)*2**104)
 return gamma
def rounding_bound(f,p,chars,late):
 """Exact all-routing enclosure: FP32 reduction, subnormal and FP64 scoring."""
 assert type(chars)==int and chars>0 and len(f)==len(p)>0
 n=len(f)
 f=[exact(x) for x in f];p=[exact(x) for x in p];late=exact(late);S=sum((max(abs(a),abs(b)) for a,b in zip(f,p)),Q(0))
 gamma=reduction_domain(n,S)
 return (gamma*S+n*Q(1,2**149))/chars+16*Q(1,2**52)*(S/chars+abs(late)+1)
def coefficients(candidate_pieces,keys,chars,late):
 assert len(candidate_pieces)==len(keys)==len(chars)==len(late)>1;allkeys=sorted({k for chain in keys for k in chain});a=[];d=[];bounds=[]
 for (f,p),chain,c,l in zip(candidate_pieces,keys,chars,late):
  assert len(f)==len(p)==len(chain)>0 and type(c)==int and c>0
  ff=[exact(v) for v in f];pp=[exact(v) for v in p];a.append(sum(ff,Q(0))/c+exact(l));row={k:Q(0) for k in allkeys}
  for k,x,y in zip(chain,ff,pp):row[k]+=(y-x)/c
  d.append([row[k] for k in allkeys]);bounds.append(rounding_bound(ff,pp,c,l))
 # Remove only columns zero for every candidate, independent of the gold.
 active=[j for j in range(len(allkeys)) if any(row[j]!=0 for row in d)];zero=[allkeys[j] for j in range(len(allkeys)) if j not in active]
 return dict(keys=[allkeys[j] for j in active],zero_keys=zero,a=a,d=[[row[j] for j in active] for row in d],B=bounds)
def margins(problem,gold):
 assert 0<=gold<len(problem['a']);wrong=[j for j in range(len(problem['a'])) if j!=gold]
 return wrong,[problem['a'][gold]-problem['a'][j] for j in wrong],[[u-v for u,v in zip(problem['d'][gold],problem['d'][j])] for j in wrong],[problem['B'][gold]+problem['B'][j] for j in wrong]
def witness_certificate(A,D,B,bits):
 assert len(A)==len(D)==len(B)>0 and len(bits)==len(D[0]) and all(type(v)==int and v in [0,1] for v in bits)
 assert all(len(row)==len(bits) for row in D)
 lower=[a+sum((v*x for v,x in zip(row,bits)),Q(0))-b for a,row,b in zip(A,D,B)];return lower,min(lower)
def normalize_dual(raw):
 assert raw and all(math.isfinite(float(x)) and float(x)>=-NEGATIVE_DUAL_CLIP for x in raw)
 v=[exact(max(float(x),0.)) for x in raw];s=sum(v,Q(0));assert s>0
 return [x/s for x in v]
def upper_certificate(A,D,B,weights):
 assert len(A)==len(D)==len(B)==len(weights)>0 and all(isinstance(w,Q) and w>=0 for w in weights) and sum(weights,Q(0))==1
 assert len({len(r) for r in D})==1
 constant=sum((w*(a+b) for w,a,b in zip(weights,A,B)),Q(0));direction=[sum((w*D[j][k] for j,w in enumerate(weights)),Q(0)) for k in range(len(D[0]))];return constant+sum((max(v,Q(0)) for v in direction),Q(0))
