"""Draft immutable artifact writes and ABCA comparisons; CPU tests only."""
import os,uuid,json,hashlib,torch
from pathlib import Path
from safetensors.torch import save_file,load_file
from integrity import compare

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write_once(path,data):
 tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
 try:
  with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.link(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()
def tensor_hash(v):
 return hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def save_capture(out,name,identity,obs,scores,logits,manifest_sha):
 if name not in ('A1','B','C','A2'):raise ValueError('name')
 out=Path(out);out.mkdir(exist_ok=True)
 tf=out/(name+'.safetensors');jf=out/(name+'.json')
 tensors={'loop0':obs['states'][0]['prefix'].contiguous(),'loop1':obs['states'][1]['prefix'].contiguous(),'rope_cos':obs['rope'][0]['cos_sin'][0].contiguous(),'rope_sin':obs['rope'][0]['cos_sin'][1].contiguous(),'positions':obs['rope'][0]['position_ids'].contiguous(),'logits':logits.contiguous()}
 tensors={k:v.clone() for k,v in tensors.items()}
 tmp=out/(name+'.'+uuid.uuid4().hex+'.partial')
 try:
  save_file(tensors,str(tmp));loaded=load_file(str(tmp))
  if set(loaded)!=set(tensors) or any(not torch.equal(loaded[k],v) or loaded[k].dtype!=v.dtype for k,v in tensors.items()):raise ValueError('readback')
  with tmp.open('rb') as f:os.fsync(f.fileno())
  os.link(tmp,tf)
 finally:
  if tmp.exists():tmp.unlink()
 attention=[]
 for a in obs['attention']:
  if a['attention_mask'] is not None:raise ValueError('mask must be none')
  attention.append({**{k:a[k] for k in ('layer_loop','past_is_none','use_cache','attention_mask')},'rope_hashes':[tensor_hash(x) for x in a['position_embeddings']]})
 payload={'name':name,'identity':identity,'scores':scores,'tensor_sha256':sha(tf),'manifest_sha256':manifest_sha,'sdpa':obs['sdpa'],'layers':obs['layers'],'attention':attention,'state_devices':[x['source_device'] for x in obs['states']],'rope_devices':obs['rope'][0]['source_devices'],'kernel':obs['kernel']}
 payload=json.loads(json.dumps(payload,sort_keys=True,allow_nan=False))
 write_once(jf,(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n').encode())
 check=json.loads(jf.read_text())
 if check!=payload or sha(tf)!=check['tensor_sha256']:raise ValueError('receipt readback')
 return check

def compare_all(out):
 out=Path(out);data={};meta={}
 for name in ('A1','B','C','A2'):
  p=out/(name+'.json');m=json.loads(p.read_text());t=out/(name+'.safetensors')
  if m['name']!=name or sha(t)!=m['tensor_sha256']:raise ValueError('capture integrity')
  data[name]=load_file(str(t));meta[name]=m
 if meta['A1']['identity']!=meta['A2']['identity'] or meta['A1']['scores']!=meta['A2']['scores']:raise ValueError('A repetition identity')
 result={}
 for name in ('A2','B','C'):
  for loop in ('loop0','loop1'):result['A1_'+name+'_'+loop]=compare(data['A1'][loop],data[name][loop])
  for key in ('rope_cos','rope_sin'):
   a=data['A1'][key];b=data[name][key];p=data['A1']['loop0'].shape[1]
   result['A1_'+name+'_'+key+'_prefix']=compare(a[:,:p,:],b[:,:p,:])
 return result
