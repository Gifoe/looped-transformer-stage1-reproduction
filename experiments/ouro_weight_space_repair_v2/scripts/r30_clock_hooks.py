"""Same-layer, same-prefix source V loop1->loop4; no spatial permutation."""
from r7_core import digest
class SourceClock:
 def __init__(self,model):
  self.modules=[l.self_attn.v_proj for l in model.model.layers];assert len(self.modules)==24;self.original=[tuple(m._forward_hooks.items()) for m in self.modules];self.handles=[];self.scorer=None;self.verified=True;self.mode=None
 def bind(self,scorer):self.scorer=scorer
 def __enter__(self):
  try:
   for i,m in enumerate(self.modules):
    def hook(module,args,output,i=i):
     import torch
     assert self.mode is not None and len(self.events)%24==i;loop=len(self.events)//24;self.events.append((loop,i));self.verified=False;assert loop<4;self.counts[i]+=1;assert output.shape[:2]==(1,self.length) and output.ndim==3 and output.dtype==torch.bfloat16;ix=torch.tensor(self.positions,device=output.device)
     if loop==0:
      self.early[i]=output.index_select(1,ix).detach().clone();self.flags.append(torch.isfinite(self.early[i]).all())
     if loop!=3 or self.mode=='capture_only':return None
     assert i in self.early and self.early[i].shape==(1,len(self.positions),output.shape[-1])
     # Identity writes the CURRENT loop4 source back. CF writes only this layer's loop1 source.
     source=output.index_select(1,ix).detach().clone() if self.mode=='identity' else self.early[i];result=output.clone();result.index_copy_(1,ix,source);self.flags.append((result.index_select(1,ix).contiguous().view(torch.uint8)==source.contiguous().view(torch.uint8)).all());mask=torch.ones(self.length,dtype=torch.bool,device=output.device);mask[ix]=False;self.flags.append((result[:,mask,:].contiguous().view(torch.uint8)==output[:,mask,:].contiguous().view(torch.uint8)).all());return result
    self.handles.append(m.register_forward_hook(hook))
   return self
  except BaseException:self.__exit__(*__import__('sys').exc_info());raise
 def configure(self,mode,tokens,context,positions):
  assert self.verified and mode in ['clock_CF','capture_only','identity'] and context in ['N1','N2'];positions=tuple(positions);assert positions and positions==tuple(sorted(set(positions))) and 0<positions[0]<=positions[-1]<len(tokens)-1
  self.mode=mode;self.context=context;self.input_sha=digest(tokens);self.positions=positions;self.length=len(tokens);self.early={};self.events=[];self.counts=[0]*24;self.flags=[];assert self.scorer is not None;self.scorer.invalidate()
 def verify(self):
  import torch
  assert self.counts==[4]*24 and len(self.events)==96 and set(self.early)==set(range(24));assert len(self.flags)==(24 if self.mode=='capture_only' else 72) and bool(torch.stack(self.flags).all().item());self.verified=True
  return dict(mode=self.mode,context=self.context,input_ids_sha256=self.input_sha,source_positions=list(self.positions),calls_per_layer=list(self.counts),events=list(self.events),capture_loop_index=0,patch_loop_index=None if self.mode=='capture_only' else 3,applied_donor_loop_index=None if self.mode=='capture_only' else (3 if self.mode=='identity' else 0),donor_layer='same_physical_layer',spatial_map='identity',other_rows_raw_exact=self.mode!='capture_only')
 def captured(self):
  assert self.verified;return tuple(self.early[i] for i in range(24))
 def __exit__(self,*args):
  errors=[]
  for h in self.handles:
   try:h.remove()
   except BaseException as e:errors.append(repr(e))
  self.handles=[]
  if self.scorer is not None:
   try:self.scorer.invalidate()
   except BaseException as e:errors.append(repr(e))
  if [tuple(m._forward_hooks.items()) for m in self.modules]!=self.original:errors.append('Original hooks not restored')
  self.cleanup=dict(handles_removed=True,original_hooks_restored=not errors,errors=errors)
  if errors and not(args and args[0] is not None):raise RuntimeError(errors)
