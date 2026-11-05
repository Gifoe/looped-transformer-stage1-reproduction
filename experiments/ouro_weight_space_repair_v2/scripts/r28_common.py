"""R28 public/source gates, atomic evidence and exact-position FP32 buffers."""
import json
from r7_core import ROOT,read,sha,digest
LOCK='configs/r28_SOURCE_PRIOR_PILOT_LOCK.json'
def save(p,x):
 p=ROOT/p;assert p.name.startswith('r28_');tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def gate():
 s=read(LOCK);assert s['status']=='frozen_bounded_development_pilot'
 for group in ['source_sha256','input_sha256']:
  for p,h in s[group].items():assert '\\' not in p and sha(p)==h,p
 r=read('logs/r28_PUBLIC_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s
RUNTIME_OUTPUTS=['results/r28_LABEL_FREE_INTEGRITY.json','results/r28_INPUT_AUDIT.json','results/r28_PROGRESS.json','results/r28_SCORES.json','results/r28_COMPLETE.json','results/r28_EXECUTION_AUDIT.json','results/r28_REPRESENTATIVE_VECTORS.npz','results/r28_NATIVE_SINGLE_POSITION_STATES.npz','results/r28_NATIVE_STATE_INDEX.json','results/r28_ANALYSIS.json','logs/r28_gpu_budget.json','logs/r28_SUPERVISOR.json','logs/r28_worker.out','logs/r28_worker.err']
def assert_fresh(paths,root=ROOT):
 existing=[p for p in paths if (root/p).exists()];assert not existing,('Preexisting output blocks this single execution',existing)
class Vectors:
 def __init__(self,timings=None,category='IO/vectors'):self.arrays={};self.timings=timings;self.category=category
 def keep(self,t):
  import torch,numpy as np,hashlib,time
  started=time.monotonic()
  assert t.dtype in [torch.bfloat16,torch.float32]
  a=t.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16) if t.dtype==torch.bfloat16 else t.detach().contiguous().cpu().numpy();dtype=str(t.dtype);shape=list(t.shape);b=a.tobytes();key=hashlib.sha256((dtype+str(tuple(shape))).encode()+b).hexdigest()
  if key in self.arrays:assert self.arrays[key].tobytes()==b
  else:self.arrays[key]=a.copy()
  if self.timings is not None:self.timings[self.category+'/capture']=self.timings.get(self.category+'/capture',0.)+time.monotonic()-started
  return dict(key=key,dtype=dtype,shape=shape,raw_sha256=hashlib.sha256(b).hexdigest())
 def write(self,path):
  import numpy as np,time
  started=time.monotonic()
  with (ROOT/path).open('xb') as f:np.savez(f,**self.arrays)
  if self.timings is not None:self.timings[self.category+'/export']=self.timings.get(self.category+'/export',0.)+time.monotonic()-started
class PositionBuffers:
 def __init__(self,rows,names,device):
  import torch
  self.buffers={};self.filled={}
  for row in rows:
   for j,c in enumerate(row['candidates']):
    for name in names:
     size=len(c['reference_target_ids']['N0']) if name.startswith('N0_') else len(c['Q_target_ids']);key=(row['id'],j,name);self.buffers[key]=torch.empty(size,dtype=torch.float32,device=device);self.filled[key]=set()
 def put(self,qid,candidate,name,position,value):
  import torch
  key=(qid,candidate,name);assert value.dtype==torch.float32 and value.numel()==1 and 0<=position<len(self.buffers[key]) and position not in self.filled[key];self.buffers[key][position]=value.reshape(());self.filled[key].add(position)
 def finish(self):
  result={}
  for key,b in self.buffers.items():
   assert self.filled[key]==set(range(len(b)))
   # One original Torch FP32 sum over the fully populated original-position buffer.
   result[key]=dict(sum_score=float(b.sum()),target_log_probabilities=b.detach().cpu().tolist())
  return result
