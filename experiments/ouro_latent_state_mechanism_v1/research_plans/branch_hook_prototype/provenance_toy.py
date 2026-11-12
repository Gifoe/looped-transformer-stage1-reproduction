"""Untrusted CPU toy provenance design, not production capture or security boundary."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,os
import torch

def digest(b):return hashlib.sha256(b).hexdigest()
def canon(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def tensor_identity(t):
 t=t.detach().contiguous();return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'sha256':digest(t.cpu().view(torch.uint8).numpy().tobytes())}
def model_identity(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(canon([k,tensor_identity(v)]))
 return h.hexdigest()
_SEALS={}  # in-memory test session only; never proof across process boundaries
@dataclass(frozen=True)
class CapturedDonor:
 seal:str
 receipt:bytes
 tensor:torch.Tensor

def validate_manifest(manifest,ids,mask,positions):
 if set(manifest)!={'question_uid','metadata_sha256','ids','mask','positions','targets','readouts'}:raise ValueError('manifest schema')
 if not isinstance(manifest['question_uid'],str) or not manifest['question_uid']:raise ValueError('question')
 if len(manifest['metadata_sha256'])!=64:raise ValueError('metadata')
 for k,v in (('ids',ids),('mask',mask),('positions',positions)):
  if manifest[k]!=tensor_identity(v):raise ValueError('manifest '+k)
 r=manifest['readouts'];n=ids.shape[-1]
 if not r or len(set(r))!=len(r) or any(type(x)!=int or not 0<=x<n for x in r):raise ValueError('readouts')
 if not manifest['targets'] or any(type(x)!=int for x in manifest['targets']):raise ValueError('targets')

def capture(model,ids,mask,positions,manifest,depth,layer_idx,source_loop,branch):
 if type(depth)!=int or depth not in (1,2) or type(source_loop)!=int or not 0<=source_loop<depth:raise ValueError('depth/loop')
 if type(layer_idx)!=int or not 0<=layer_idx<len(model.layers) or branch not in ('attention','mlp'):raise ValueError('site')
 validate_manifest(manifest,ids,mask,positions)
 if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):raise ValueError('unregistered hook')
 target=model.layers[layer_idx];site=target.input_layernorm_2 if branch=='attention' else target.post_attention_layernorm_2
 if target.self_attn.layer_idx!=layer_idx:raise ValueError('layer')
 before_model=model_identity(model); before_inputs=[tensor_identity(v) for v in (ids,mask,positions)]; before_manifest=digest(canon(manifest))
 events=[];pending=[];current={'ut':None};handles=[]
 for i,l in enumerate(model.layers):
  def before(m,args,kwargs,i=i):
   u=kwargs.get('current_ut');
   if type(u)!=int:raise ValueError('ut')
   events.append((u,i));current['ut']=u
  handles.append(l.register_forward_pre_hook(before,with_kwargs=True))
 def collect(m,args,out):
  if current['ut']==source_loop:
   if not torch.isfinite(out).all():raise ValueError('nonfinite')
   pending.append(out.detach().clone())
 handles.append(site.register_forward_hook(collect))
 try:
  result=model(ids,mask,positions,depth)
  if events!=[(u,i) for u in range(depth) for i in range(len(model.layers))] or len(pending)!=1 or not torch.is_tensor(result) or not torch.isfinite(result).all():raise ValueError('incomplete/nonfinite forward')
  if before_model!=model_identity(model) or before_inputs!=[tensor_identity(v) for v in (ids,mask,positions)] or before_manifest!=digest(canon(manifest)):raise ValueError('forward mutation')
 finally:
  for h in handles:h.remove()
 receipt={'schema':'toy-captured-v1','question_uid':manifest['question_uid'],'manifest_sha256':digest(canon(manifest)),'metadata_sha256':manifest['metadata_sha256'],'inputs':{k:tensor_identity(v) for k,v in (('ids',ids),('mask',mask),('positions',positions))},'targets':manifest['targets'],'readouts':manifest['readouts'],'model_sha256':model_identity(model),'depth':depth,'source_loop':source_loop,'source_kind':'independent_K1' if depth==1 and source_loop==0 else 'K2_run_first_pass' if depth==2 and source_loop==0 else 'other','layer':layer_idx,'branch':branch,'module_path':f'layers.{layer_idx}.'+('input_layernorm_2' if branch=='attention' else 'post_attention_layernorm_2'),'tensor':tensor_identity(pending[0]),'events_sha256':digest(canon(events)),'forward_complete':True,'receiver_state':'natural','lineage':[]}
 blob=canon(receipt);seal=os.urandom(32).hex();_SEALS[seal]=digest(blob);return CapturedDonor(seal,blob,pending[0]),result

def verify(donor,model,ids,mask,positions,manifest,layer_idx,branch,allowed_source,receiver_lineage,identity=False):
 if not isinstance(donor,CapturedDonor) or _SEALS.get(donor.seal)!=digest(donor.receipt):raise ValueError('not sealed')
 r=json.loads(donor.receipt);validate_manifest(manifest,ids,mask,positions)
 if identity:raise NotImplementedError('identity requires controlled receiver lineage')
 if r['inputs']!={k:tensor_identity(v) for k,v in (('ids',ids),('mask',mask),('positions',positions))}:raise ValueError('receiver inputs')
 if not r.get('forward_complete') or r['tensor']!=tensor_identity(donor.tensor):raise ValueError('tensor/receipt')
 checks={'question_uid':manifest['question_uid'],'manifest_sha256':digest(canon(manifest)),'model_sha256':model_identity(model),'layer':layer_idx,'branch':branch}
 if any(r.get(k)!=v for k,v in checks.items()):raise ValueError('shared identity')
 if (r['depth'],r['source_loop'])!=allowed_source:raise ValueError('source policy')
 if identity and (r['receiver_state']!='hybrid' or r['lineage']!=receiver_lineage):raise ValueError('identity lineage')
 return donor.tensor

def patch_forward(donor,model,ids,mask,positions,manifest,depth,layer_idx,receiver_loop,branch,region,allowed_source):
 """Controlled natural receiver toy only; hybrid and identity remain unsupported."""
 if type(depth)!=int or depth!=2 or type(receiver_loop)!=int or receiver_loop!=1:raise ValueError('receiver depth/loop')
 validate_manifest(manifest,ids,mask,positions)
 if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):raise ValueError('unregistered hook')
 n=ids.shape[-1];readouts=manifest['readouts'];idx=readouts if region=='R' else [i for i in range(n) if i not in readouts] if region=='C' else None
 if not idx:raise ValueError('region')
 before_model=model_identity(model);before_inputs=[tensor_identity(v) for v in (ids,mask,positions)];before_manifest=digest(canon(manifest))
 tensor=verify(donor,model,ids,mask,positions,manifest,layer_idx,branch,allowed_source,[]).detach().clone();tensor_before=tensor_identity(tensor)
 if tensor_before!=json.loads(donor.receipt)['tensor']:raise ValueError('donor snapshot')
 target=model.layers[layer_idx]
 if target.self_attn.layer_idx!=layer_idx:raise ValueError('receiver layer')
 site=target.input_layernorm_2 if branch=='attention' else target.post_attention_layernorm_2
 events=[];visits=[];current={'ut':None};handles=[]
 for i,l in enumerate(model.layers):
  def before(m,args,kwargs,i=i):
   u=kwargs.get('current_ut')
   if type(u)!=int:raise ValueError('ut')
   events.append((u,i));current['ut']=u
  handles.append(l.register_forward_pre_hook(before,with_kwargs=True))
 def inject(m,args,out):
  if current['ut']!=receiver_loop:return None
  if tensor_identity(tensor)!=tensor_before:raise ValueError('private donor mutated')
  if visits or not torch.isfinite(out).all() or out.shape!=tensor.shape or out.dtype!=tensor.dtype or out.device!=tensor.device:raise ValueError('receiver shape/finite/visits')
  visits.append(1);z=out.clone();z[:,idx,:]=tensor[:,idx,:];return z
 handles.append(site.register_forward_hook(inject))
 try:
  result=model(ids,mask,positions,depth)
  if events!=[(u,i) for u in range(depth) for i in range(len(model.layers))] or len(visits)!=1 or not torch.isfinite(result).all():raise ValueError('receiver incomplete')
  if before_model!=model_identity(model) or before_inputs!=[tensor_identity(v) for v in (ids,mask,positions)] or before_manifest!=digest(canon(manifest)):raise ValueError('receiver mutation')
 finally:
  for h in handles:h.remove()
 return result
