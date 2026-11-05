"""Synthetic numeric and hook checks, no real model or ARC outcome input."""
import numpy as np,torch,types,json,ast,hashlib
from pathlib import Path
from r28_math import numpy_readouts,torch_readouts
from r28_hooks import AnswerReplay,NativeDonor
ROOT=Path(__file__).resolve().parents[1]
def toy():return types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(v_proj=torch.nn.Identity())) for _ in range(24)]))
class Guard:
 def __init__(self):self.n=0
 def invalidate(self):self.n+=1

def run(c,x):
 out=[]
 for loop in range(c.loops):
  for i,m in enumerate(c.modules):out.append((loop,i,m(x)))
 c.verify();return out

def main():
 checks=[];rng=np.random.default_rng(2021);maxerror=0.;maxdiag=0.
 for _ in range(30):
  V=37;q4=rng.normal(size=V).astype('float32');n1=rng.normal(size=(3,V)).astype('float32');n4=rng.normal(size=(3,V)).astype('float32');b4=rng.normal(size=(2,V)).astype('float32');q1=np.float32(q4-2*np.float32(.5*(n4[1:]-b4)).mean(axis=0));perm=rng.permutation(V)
  a,d=numpy_readouts(q1,q4,n1,n4,b4,perm);b,e=torch_readouts(*[torch.from_numpy(x) for x in [q1,q4,n1,n4,b4]],torch.from_numpy(perm))
  for k in a:
   error=float(np.max(np.abs(a[k]-b[k].numpy())));maxerror=max(maxerror,error);assert error<=1e-6
  for k in ['alpha','numerator','denominator','main_norm2','G_norm_error','null_norm_error']:
   error=float(np.max(np.abs(np.asarray(d[k])-e[k].numpy())));maxdiag=max(maxdiag,error);assert error<=1e-10
  zero,zd=numpy_readouts(q1,q4,n1,n4,n4[1:].copy(),perm);assert np.array_equal(zero['primary'],zero['Fixed']) and zd['alpha']==0 and not zd['G_unmatched'] and not zd['null_unmatched']
  changed1=n1.copy();changed4=n4.copy();changed1[0]+=32;changed4[0]-=16;changed,_=numpy_readouts(q1,q4,changed1,changed4,b4,perm);assert np.array_equal(changed['primary'],a['primary'])
  swapped,_=numpy_readouts(q1,q4,n1[[0,2,1]],n4[[0,2,1]],b4[::-1],perm);assert np.max(np.abs(swapped['primary']-a['primary']))<=1e-6
  for unmatched,error in [(d['G_unmatched'],d['G_norm_error']),(d['null_unmatched'],d['null_norm_error'])]:
   if not unmatched:assert error<=1e-10
 checks.append('independent NumPy/Torch/fullV robust projection/normmatch/zeroFixed/noN0/main reference permutation')
 bad1=n4.copy();unmatched,ud=numpy_readouts(q1,q4,bad1,n4,b4,perm);assert ud['main_norm2']>0 and ud['G_unmatched']
 try:numpy_readouts(q1*np.nan,q4,n1,n4,b4,perm)
 except ValueError:pass
 else:raise AssertionError('nonfinite accepted')
 checks.append('raw G direction zero with nonzero main is unmatched/no epsilon/nonfinite rejects')
 x=torch.arange(64).reshape(1,16,4).to(torch.bfloat16);positions=(1,2,4,5,7,8);permutation=(2,3,4,5,0,1);tokens=list(range(16));model=toy();g=Guard()
 with AnswerReplay(model) as c:
  c.bind(g);c.configure('capture',tokens,'N1',positions,permutation);out=run(c,x);assert all(y is x for _,_,y in out);native=c.take_donor()
  for i in range(24):assert torch.equal(native.values[i],x[:,positions])
  c.configure('patch',tokens,'N1',positions,permutation,native);out=run(c,x)
  for loop,i,y in out:
   if loop<3:assert y is x
   else:
    assert y is not x and torch.equal(y[:,positions],x[:,[positions[j] for j in permutation]])
    other=[i for i in range(16) if i not in positions];assert torch.equal(y[:,other],x[:,other])
  try:c.take_donor()
  except AssertionError:pass
  else:raise AssertionError('CF donor allowed')
  try:c.configure('patch',tokens+[1],'N1',positions,permutation,native)
  except AssertionError:pass
  else:raise AssertionError('different-length donor allowed')
  c.configure('patch',tokens,'N1',positions,tuple(range(6)),native);out=run(c,x);assert all(torch.equal(y,x) for _,_,y in out)
  c.configure('off',tokens,'Q',loops=5);out=run(c,x);assert all(y is x for _,_,y in out)
 assert c.cleanup['original_hooks_restored'] and all(not m._forward_hooks for m in c.modules)
 checks.append('native-only clamped block replay/within-block order/cycle multiset/identity/other rows/5loop off/noop cleanup')
 with AnswerReplay(toy()) as c:
  c.bind(Guard());c.configure('capture',tokens,'N1',positions,permutation);c.modules[0](x)
  try:c.configure('off',tokens,'Q',loops=5)
  except AssertionError:pass
  else:raise AssertionError('unverified forward overwritten')
  try:c.verify()
  except AssertionError:pass
  else:raise AssertionError('partial clock accepted')
 model=toy();controller=AnswerReplay(model)
 def fail(*args,**kwargs):raise RuntimeError('synthetic register failure')
 model.model.layers[4].self_attn.v_proj.register_forward_hook=fail
 try:controller.__enter__()
 except RuntimeError:pass
 else:raise AssertionError('partial register accepted')
 assert controller.cleanup['original_hooks_restored'] and not controller.handles
 try:
  with AnswerReplay(toy()) as c:c.bind(Guard());c.configure('off',tokens,'Q',loops=5);raise ValueError('synthetic original failure')
 except ValueError:pass
 assert c.cleanup['original_hooks_restored']
 checks.append('wrong clock/unverified configure/partial registration/original exception cleanup')
 for p in ['scripts/r28_math.py','scripts/r28_hooks.py']:ast.parse((ROOT/p).read_text())
 receipt=dict(status='passed_CPU_synthetic',checks=checks,max_FP32_logits_Numpy_Torch_error=maxerror,max_FP64_diag_error=maxdiag,model_loaded=False,GPU_worker_seconds=0,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/r28_math.py','scripts/r28_hooks.py','scripts/r28_cpu_checks.py']})
 (ROOT/'results/r28_CPU_MATH_HOOK_CHECKS.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
