"""L18 natural swap controlled single question; draft, no standalone entrypoint."""
import torch,hashlib,json
from natural_l18_hook import intervene,matched_norm
from natural_sources import load as load_source
from technical import verify,score
from state_capture import bits
if not __debug__:raise RuntimeError('optimized Python forbidden')
@torch.inference_mode()
def question(model,split,ident,guard):
 if model.training or model.config._attn_implementation!='sdpa':raise ValueError('eval / sdpa')
 guard();src=load_source(split,ident);e=src['natural'];device=next(model.parameters()).device
 runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device)}
 if any(e['runtime'][k]!=runtime[k] for k in runtime) or e['runtime']['attention']!='sdpa' or e['runtime']['dtype']!='bfloat16' or e['runtime']['use_cache'] is not False:raise ValueError('original runtime mismatch')
 if device.type!='cuda':raise ValueError('production requires CUDA')
 ids=torch.tensor([e['ids']],device=device,dtype=torch.long);R=e['R'];targets=e['targets'];core=model.model;old=core.total_ut_steps
 try:
  obs,base=verify(model,ids,R,targets,e['states'],e['norms'],e['logits'],e['scores'])
  guard()
  for u in (0,1):
   for name in ('x','a'):
    if not bits(obs['states'][(u,17)][name],src['states'][u][name].to(device)):raise ValueError('natural saved donor/receiver mismatch')
  if any(not bits(base[u],e['logits'][u].to(device)) or score(base[u],targets)!=e['scores'][u] for u in (0,1)):raise ValueError('baseline')
  results={'B1':base[0].detach().clone(),'B2':base[1].detach().clone()};controls={}
  def run(depth,donor):
   guard();u=depth-1;core.total_ut_steps=depth;own=src['states'][u]
   with intervene(model,depth,R,own['x'],own['a'],donor) as record:
    y=model(ids,labels=None,use_cache=False,exit_at_step=depth-1)
   z=y.logits[0,R,:].unsqueeze(0)
   if z.dtype!=torch.bfloat16 or z.shape!=(1,49152) or not torch.isfinite(z).all():raise ValueError('readout')
   if record['visits']!=1:raise ValueError('missing site')
   guard()
   return z.detach().clone()
  a1=src['states'][0]['a'].to(device);a2=src['states'][1]['a'].to(device)
  results['I1']=run(1,a1);results['I2']=run(2,a2)
  if not bits(results['I1'],results['B1']) or not bits(results['I2'],results['B2']):raise ValueError('identity')
  results['T21']=run(2,a1);results['T12']=run(1,a2)
  if bits(a1,a2) and (not bits(results['T21'],results['B2']) or not bits(results['T12'],results['B1'])):raise ValueError('equal donor anchor')
  for name,depth,donor,receiver in [('T21_norm',2,a1,a2),('T12_norm',1,a2,a1)]:
   normalized,record=matched_norm(donor,receiver);controls[name]=record
   if normalized is not None:results[name]=run(depth,normalized)
  scores={k:score(z,targets) for k,z in results.items()}
  if scores['B1']!=e['scores'][0] or scores['B2']!=e['scores'][1] or scores['I1']!=scores['B1'] or scores['I2']!=scores['B2']:raise ValueError('score anchor')
  guard()
  state_sha={str(u):{k:hashlib.sha256(v.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in src['states'][u].items()} for u in (0,1)}
  return {'input_ids_sha256':hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest(),'state_sha256':state_sha,'id':ident,'split':split,'targets':targets,'gold':e['gold'],'R':R,'source_metadata_sha256':src['source_metadata_sha256'],'source_tensor_sha256':src['source_tensor_sha256'],'natural_metadata_sha256':src['natural_metadata_sha256'],'natural_tensor_sha256':src['natural_tensor_sha256'],'combined_sha256':src['combined_sha256'],'controls':controls,'scores':scores,'logits':results}
 finally:core.total_ut_steps=old
