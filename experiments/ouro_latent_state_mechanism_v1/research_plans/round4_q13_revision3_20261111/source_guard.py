"""Local CPU staging, content-bound inputs and model; not production approved."""
import hashlib,json,torch
from pathlib import Path

def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def tensor_sha(t):
 return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def state_digest(model):
 h=hashlib.sha256()
 for name,t in sorted(model.state_dict().items()):
  h.update(json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(tensor_sha(t).encode())
 return h.hexdigest()
class FrozenSources:
 """Trusted-code drift detection, not protection from transient malicious changes."""
 def __init__(self,model,ids,paths,expected):
  self.model=model;self.ids=ids;self.paths=tuple(map(Path,paths));self.baseline=json.loads(json.dumps(expected))
  if self.snapshot()!=self.baseline:raise ValueError('initial provenance mismatch')
  self.quick_baseline=self.quick()
 def snapshot(self):
  return {'model':state_digest(self.model),'ids':tensor_sha(self.ids),'files':{str(p.resolve()):sha_file(p) for p in self.paths}}
 def quick(self):
  tensors=list(self.model.named_parameters())+list(self.model.named_buffers())
  params=[(name,id(t),t.data_ptr(),t._version,str(t.dtype),str(t.device),tuple(t.shape)) for name,t in tensors]
  return {'params':params,'ids':tensor_sha(self.ids),'files':[(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in self.paths]}
 def __call__(self):
  if self.quick()!=self.quick_baseline:raise ValueError('model/input/source drift')
 def finish(self):
  self()
  if self.snapshot()!=self.baseline:raise ValueError('final full provenance mismatch')
