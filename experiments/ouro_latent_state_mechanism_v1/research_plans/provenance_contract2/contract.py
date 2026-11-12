"""CPU synthetic design, not a production evidence/security boundary."""
import hashlib,json,os
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file

def digest(x):return hashlib.sha256(x).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def signature(t):
 t=t.detach().contiguous();return {'dtype':str(t.dtype),'shape':list(t.shape),'device':str(t.device),'sha256':digest(t.cpu().view(torch.uint8).numpy().tobytes())}
def weights(model):return digest(canonical({k:signature(v) for k,v in sorted(model.state_dict().items())}))
def effective(mask,positions):
 n=mask.shape[-1]; causality=torch.tril(torch.ones(n,n,dtype=torch.bool));allowed=causality & mask.bool()[:,None,:];angle=positions.float().unsqueeze(-1)*torch.tensor([.1,.3]);return allowed,torch.cos(angle),torch.sin(angle)
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.emb=torch.nn.Embedding(32,4);self.layers=torch.nn.ModuleList([torch.nn.Linear(4,4,bias=False) for _ in range(3)]);self.norm=torch.nn.LayerNorm(4)
 def forward(self,ids,mask,positions,depth,session,action=None,fail=False):
  allowed,cos,sin=effective(mask,positions);h=self.emb(ids);session['effective']={'mask':signature(allowed),'cos':signature(cos),'sin':signature(sin)}
  for u in range(depth):
   if u==1 and action:
    before=signature(h);h=h.clone();h[:,action['position'],:]+=action['delta'];session['lineage'].append({'kind':'add','loop':u,'position':action['position'],'delta':action['delta'],'before':before,'after':signature(h)})
   for i,layer in enumerate(self.layers):
    session['events'].append([u,i]);pairs=h.reshape(1,h.shape[1],2,2);rot=torch.stack((pairs[...,0]*cos-pairs[...,1]*sin,pairs[...,0]*sin+pairs[...,1]*cos),-1).reshape_as(h);scores=rot@rot.transpose(-1,-2);attention=torch.softmax(scores.masked_fill(~allowed,float('-inf')),dim=-1)@h
    inc=layer(attention);h=h+inc
    if (u,i)==session['site']:
     session['pending']=inc.detach().clone();session['capture_lineage']=json.loads(json.dumps(session['lineage']))
   h=self.norm(h)
  if fail:raise RuntimeError('toy injected failure')
  return h

def execute(model,ids,mask,positions,manifest,depth,site,action=None,fail=False):
 if depth not in (1,2) or type(depth)!=int or site[0]>=depth or site[1]>=len(model.layers):raise ValueError('site')
 if manifest['ids']!=signature(ids) or manifest['mask']!=signature(mask) or manifest['positions']!=signature(positions):raise ValueError('manifest')
 if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):raise ValueError('unregistered hook')
 pre=(weights(model),*[signature(t) for t in (ids,mask,positions)],digest(canonical(manifest)));s={'status':'PENDING','site':site,'pending':None,'events':[],'lineage':[]}
 output=model(ids,mask,positions,depth,s,action,fail)
 if s['events']!=[[u,i] for u in range(depth) for i in range(len(model.layers))] or s['pending'] is None or not torch.isfinite(output).all():raise ValueError('incomplete')
 post=(weights(model),*[signature(t) for t in (ids,mask,positions)],digest(canonical(manifest)))
 if pre!=post:raise ValueError('changed inputs or weights')
 r={'status':'SEALED','kind':'toy','manifest_sha256':pre[-1],'model_sha256':pre[0],'inputs':{'ids':pre[1],'mask':pre[2],'positions':pre[3]},'effective':s['effective'],'depth':depth,'site':list(site),'capture_lineage':s['capture_lineage'],'forward_lineage':s['lineage'],'receiver':'hybrid' if s['capture_lineage'] else 'natural','events_sha256':digest(canonical(s['events'])),'tensor':signature(s['pending'])}
 return r,s['pending'],output

def write_pair(directory,receipt,tensor):
 p=Path(directory);p.mkdir(exist_ok=False);temp=p/'tensor.partial';save_file({'donor':tensor},str(temp));os.link(temp,p/'tensor.safetensors');temp.unlink();raw=canonical(receipt);f=p/'receipt.json'
 with f.open('xb') as h:h.write(raw);h.flush();os.fsync(h.fileno())
 return digest(raw)
def read_pair(directory,expected_receipt_hash):
 p=Path(directory);raw=(p/'receipt.json').read_bytes()
 if digest(raw)!=expected_receipt_hash:raise ValueError('receipt hash')
 receipt=json.loads(raw);items=load_file(str(p/'tensor.safetensors'))
 if receipt.get('status')!='SEALED' or set(items)!={'donor'} or signature(items['donor'])!=receipt['tensor']:raise ValueError('tensor receipt')
 return receipt,items['donor']
def identity_compatible(a,b):
 keys=('manifest_sha256','model_sha256','inputs','effective','depth','site','capture_lineage','receiver')
 return all(a[k]==b[k] for k in keys)
