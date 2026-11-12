"""Local staging: immutable 12-logit + 4-branch artifacts and replay."""
import hashlib,json,os,uuid,torch
from pathlib import Path
from safetensors.torch import save_file,load_file

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def score(z,targets,device):
 if z.dtype!=torch.bfloat16 or z.ndim!=2 or z.shape[0]!=1 or not torch.isfinite(z).all():raise ValueError('logits')
 if not targets or any(type(t)!=int or t<0 or t>=z.shape[1] for t in targets):raise ValueError('targets')
 z=z.to(device);return [float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=device)).sum()) for t in targets]
def save(out,identity,results,targets,readout,expected_vocab,expected_hidden,source_guard):
 out=Path(out);out.mkdir(exist_ok=False);source_guard();tensors={};scores={}
 for r in ('1','2'):
  pair=results[r]
  for key in ('B1','B2','I1','I2','U1','U2'):
   z=pair[key]
   if tuple(z.shape)!=(1,expected_vocab):raise ValueError('vocabulary')
   tensors[key+'r'+r]=z.detach().cpu().contiguous().clone();scores[key+'r'+r]=score(z,targets,z.device)
  for i in ('1','2'):
   a=pair['capture'+i].branch[:,readout,:]
   if a.dtype!=torch.bfloat16 or tuple(a.shape)!=(1,len(readout),expected_hidden) or not torch.isfinite(a).all():raise ValueError('branch')
   tensors['a'+i+'r'+r]=a.detach().cpu().contiguous().clone()
 temp=out/('values.'+uuid.uuid4().hex+'.partial');final=out/'values.safetensors'
 try:
  save_file(tensors,str(temp));check=load_file(str(temp))
  if set(check)!=set(tensors) or any(not torch.equal(check[k],tensors[k]) for k in tensors):raise ValueError('tensor readback')
  with temp.open('r+b') as f:os.fsync(f.fileno())
  os.link(temp,final)
 finally:
  if temp.exists():temp.unlink()
 payload={'identity':identity,'targets':targets,'readout':readout,'vocab_size':expected_vocab,'hidden_size':expected_hidden,'scores':scores,'tensor_sha256':sha(final)}
 envelope={'payload':payload,'payload_sha256':hashlib.sha256(canonical(payload)).hexdigest()};source_guard()
 m=out/'metadata.json';tmp=out/('metadata.'+uuid.uuid4().hex+'.partial')
 try:
  with tmp.open('xb') as f:f.write(canonical(envelope));f.flush();os.fsync(f.fileno())
  os.link(tmp,m)
 finally:
  if tmp.exists():tmp.unlink()
 return envelope

def replay(out,identity,targets,readout,expected_vocab,expected_hidden,device,source_guard):
 out=Path(out);source_guard();m=json.loads((out/'metadata.json').read_text());v=m['payload']
 if m['payload_sha256']!=hashlib.sha256(canonical(v)).hexdigest() or v['identity']!=identity or v['targets']!=targets or v['readout']!=readout or v['vocab_size']!=expected_vocab or v['hidden_size']!=expected_hidden or v['tensor_sha256']!=sha(out/'values.safetensors'):raise ValueError('provenance')
 t=load_file(str(out/'values.safetensors'));logits={k+'r'+r for k in ('B1','B2','I1','I2','U1','U2') for r in ('1','2')};branches={'a'+i+'r'+r for i in ('1','2') for r in ('1','2')}
 if set(t)!=logits|branches or set(v['scores'])!=logits:raise ValueError('tensor keys')
 for k in logits:
  if tuple(t[k].shape)!=(1,expected_vocab) or score(t[k],targets,device)!=v['scores'][k]:raise ValueError('score replay')
 for k in branches:
  if t[k].dtype!=torch.bfloat16 or tuple(t[k].shape)!=(1,len(readout),expected_hidden) or not torch.isfinite(t[k]).all():raise ValueError('branch shape')
 source_guard();return {'metadata_sha256':sha(out/'metadata.json'),'tensor_sha256':sha(out/'values.safetensors'),'device':str(device),'status':'ZERO_SCORE_DIFFERENCE'}
