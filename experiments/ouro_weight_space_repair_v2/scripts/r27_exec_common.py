"""R27 execution namespace, immutable public gates."""
import json
from r7_core import ROOT,read,sha,digest
LOCK='configs/r27_EXECUTION_LOCK.json'
def save(p,x):
 p=ROOT/p;assert p.name.startswith('r27_');tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def gate():
 s=read(LOCK);assert s['status']=='frozen_bounded_functional_diagnostic'
 for group in ['source_sha256','input_sha256']:
  for p,h in s[group].items():assert '\\' not in p and sha(p)==h,p
 r=read('logs/r27_PUBLIC_EXECUTION_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s

class Vectors:
 def __init__(self):self.arrays={}
 def keep(self,t):
  import torch,numpy as np,hashlib
  assert t.dtype in [torch.bfloat16,torch.float32]
  a=t.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16) if t.dtype==torch.bfloat16 else t.detach().contiguous().cpu().numpy()
  dtype=str(t.dtype);shape=list(t.shape);raw=a.tobytes();h=hashlib.sha256(raw).hexdigest();key=hashlib.sha256((dtype+str(tuple(shape))).encode()+raw).hexdigest()
  if key in self.arrays:assert self.arrays[key].tobytes()==raw
  else:self.arrays[key]=a.copy()
  return dict(key=key,dtype=dtype,shape=shape,raw_sha256=h)
 def write(self):
  import numpy as np
  with (ROOT/'results/r27_EXEC_VECTORS.npz').open('xb') as f:np.savez(f,**self.arrays)
