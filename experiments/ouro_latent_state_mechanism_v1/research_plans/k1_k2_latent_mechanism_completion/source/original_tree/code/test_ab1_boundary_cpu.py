"""Synthetic 24-layer causal oracle; no checkpoint, subject, or GPU."""
import torch
from ab1_boundary_core import corner,q9_cross,control

torch.manual_seed(2114);T=8;P=3;R=T-1;D=6;E=torch.randn(1,T,D,dtype=torch.float64);ws=[torch.randn(D,D,dtype=torch.float64)*.04 for _ in range(24)];head=torch.randn(D,5,dtype=torch.float64)
def norm(x):return x / torch.sqrt(x.square().mean(-1,keepdim=True)+1e-5)
def run(h,start=0,stop=24):
 for j in range(start,stop):
  causal=torch.cumsum(h,dim=1)/torch.arange(1,T+1,dtype=h.dtype)[None,:,None];h=h+torch.tanh(causal@ws[j])*.1
 return h
K1=run(E);B=norm(K1);native=run(B);assert torch.equal(norm(run(E)),norm(K1))
# Full E-all replay uses same positional dynamics, not prefix-retaining H00.
assert torch.equal(norm(run(E))[:,R,:]@head,norm(K1)[:,R,:]@head)
H={a+b:corner(E,B,P,R,a,b) for a in ('E','B') for b in ('E','B')};assert torch.equal(H['BB'],B) and torch.equal(H['EE'][:,:P],B[:,:P]) and not torch.equal(H['EE'],E)
assert torch.equal(norm(run(H['BB']))[:,R,:]@head,norm(native)[:,R,:]@head)
for a in ('E','B'):
 for b in ('E','B'):
  x=H[a+b];assert torch.equal(x[:,:P],B[:,:P]);assert torch.equal(x[:,P:R],(E if a=='E' else B)[:,P:R]);assert torch.equal(x[:,R],(E if b=='E' else B)[:,R])
N9=run(H['BB'],0,8);U9=run(H['EB'],0,8);X=q9_cross(U9,N9,P,R);assert torch.equal(X[:,P:R],N9[:,P:R]) and torch.equal(X[:,:P],U9[:,:P]) and torch.equal(X[:,R],U9[:,R]);Y=q9_cross(N9,U9,P,R);assert torch.equal(Y[:,P:R],U9[:,P:R]);run(X,8,24);run(Y,8,24)
for region in ('Q','R'):
 a=H['EB'];b=H['BB'];out,receipt=control(a,b,P,R,region,'random','train','sample','edge1');out2,rr=control(a,b,P,R,region,'random','train','sample','edge1');assert torch.equal(out,out2) and receipt['status']=='OK';assert all(abs(r['nominal_displacement']-r['actual_displacement'])<1e-12 for r in receipt['rows'])
 out,receipt=control(a,b,P,R,region,'norm','train','sample','edge1');assert receipt['status']=='OK';assert all(abs(r['actual_norm']-r['receiver_norm'])<1e-12 for r in receipt['rows'])
zero=torch.zeros_like(E);one=torch.ones_like(E);out,r=control(one,zero,P,R,'Q','norm','train','x','q');assert out is None and r['status']=='UNDEFINED_ZERO_DONOR';out,r=control(zero,zero,P,R,'R','norm','train','x','q');assert torch.equal(out,zero)
print('PASS synthetic 24-layer causal boundary, E-all identity, B->K2, four corners, Q9 cross, per-token controls, finite and zero branches; not model verification')
