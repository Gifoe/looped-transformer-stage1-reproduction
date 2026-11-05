"""Synthetic clock/lifecycle tests only; no real model or task outcomes."""
import types,json,hashlib
from pathlib import Path
import torch
from r30_clock_hooks import SourceClock
ROOT=Path(__file__).resolve().parents[1]
class Guard:
 def __init__(self):self.n=0
 def invalidate(self):self.n+=1
def toy():return types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(v_proj=torch.nn.Identity())) for _ in range(24)]))
def raw(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def fail(f):
 try:f()
 except AssertionError:return
 raise AssertionError('Negative fixture unexpectedly accepted')
def forward(c,mode):
 c.configure(mode,list(range(9)),'N1',[2,4,6]);outs=[]
 for loop in range(4):
  for layer,m in enumerate(c.modules):
   x=(torch.arange(72).reshape(1,9,8)+1000*loop+100*layer).to(torch.bfloat16);y=m(x);outs.append(y)
   if loop<3 or mode=='capture_only':assert y is x
   else:
    assert y is not x and raw(y[:,[0,1,3,5,7,8]],x[:,[0,1,3,5,7,8]])
    expected=x if mode=='identity' else (torch.arange(72).reshape(1,9,8)+100*layer).to(torch.bfloat16)
    assert raw(y[:,[2,4,6]],expected[:,[2,4,6]])
 audit=c.verify();assert audit['applied_donor_loop_index']==(None if mode=='capture_only' else 3 if mode=='identity' else 0);assert len(c.captured())==24
 return audit
def main():
 model=toy();guard=Guard();sentinel=model.model.layers[0].self_attn.v_proj.register_forward_hook(lambda *args:None);original=[tuple(l.self_attn.v_proj._forward_hooks.items()) for l in model.model.layers];checks=[]
 with SourceClock(model) as c:
  c.bind(guard)
  for mode in ['capture_only','identity','clock_CF']:forward(c,mode);checks.append(mode+'_exact')
  c.configure('clock_CF',list(range(9)),'N2',[2,4,6]);c.modules[0](torch.ones(1,9,8,dtype=torch.bfloat16));fail(lambda:c.configure('identity',list(range(9)),'N2',[2,4,6]));fail(c.verify);checks+=['unverified_reconfigure_rejected','incomplete_count_rejected']
 assert [tuple(l.self_attn.v_proj._forward_hooks.items()) for l in model.model.layers]==original
 try:
  with SourceClock(model) as c:
   c.bind(guard);c.configure('clock_CF',list(range(9)),'N1',[2,4,6]);c.modules[1](torch.ones(1,9,8,dtype=torch.bfloat16))
 except AssertionError:pass
 else:raise AssertionError('wrong layer order accepted')
 assert [tuple(l.self_attn.v_proj._forward_hooks.items()) for l in model.model.layers]==original;checks+=['wrong_layer_order_rejected','exception_cleanup_original_hooks_restored']
 sentinel.remove();assert guard.n==7
 result=dict(status='passed',checks=checks,torch=torch.__version__,CPU_only=True,model_loaded=False,GPU_worker_seconds=0,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/r30_clock_hooks.py','scripts/r30_cpu_checks.py']})
 with (ROOT/'results/r30_CPU_CLOCK_CHECKS.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
