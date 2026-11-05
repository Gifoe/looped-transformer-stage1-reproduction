"""Native-source-clamped last-loop whole-answer V replay; no CF donor capture."""
from dataclasses import dataclass
from r7_core import digest
@dataclass(frozen=True)
class NativeDonor:
 input_sha256:str
 context:str
 positions:tuple
 values:tuple
 origin:str='native_capture'
class AnswerReplay:
 def __init__(self,model):
  self.modules=[l.self_attn.v_proj for l in model.model.layers];assert len(self.modules)==24;self.original=[tuple(m._forward_hooks.items()) for m in self.modules];self.handles=[];self.scorer=None;self.verified=True;self.mode=None
 def bind(self,scorer):self.scorer=scorer
 def __enter__(self):
  try:
   for i,m in enumerate(self.modules):
    def hook(module,args,output,i=i):
     import torch
     assert self.mode is not None and len(self.events)%24==i;loop=len(self.events)//24;self.events.append((loop,i));self.counts[i]+=1;self.verified=False;assert loop<self.loops and self.counts[i]<=self.loops
     assert output.ndim==3 and output.shape[:2]==(1,self.length) and output.dtype==torch.bfloat16
     if self.mode=='off' or loop!=3:return None
     ix=torch.tensor(self.positions,device=output.device)
     if self.mode=='capture':
      self.captured[i]=output.index_select(1,ix).detach().clone();self.flags.append(torch.isfinite(self.captured[i]).all());return None
     native=self.donor.values[i];assert native.shape==(1,len(self.positions),output.shape[-1]) and native.dtype==output.dtype and native.device==output.device
     source=native.index_select(1,torch.tensor(self.permutation,device=output.device));result=output.clone();result.index_copy_(1,ix,source);self.flags.append((result.index_select(1,ix).contiguous().view(torch.uint8)==source.contiguous().view(torch.uint8)).all());mask=torch.ones(self.length,dtype=torch.bool,device=output.device);mask[ix]=False;self.flags.append((result[:,mask,:].contiguous().view(torch.uint8)==output[:,mask,:].contiguous().view(torch.uint8)).all());return result
    self.handles.append(m.register_forward_hook(hook))
   return self
  except BaseException:self.__exit__(*__import__('sys').exc_info());raise
 def configure(self,mode,tokens,context,positions=(),permutation=(),donor=None,loops=4):
  assert self.verified and mode in ['off','capture','patch'] and loops in [4,5]
  key=digest(tokens);positions=tuple(positions);permutation=tuple(permutation)
  if mode!='off':assert loops==4 and positions==tuple(sorted(set(positions))) and positions and 0<min(positions)<=max(positions)<len(tokens)-1 and sorted(permutation)==list(range(len(positions)))
  if mode=='patch':assert isinstance(donor,NativeDonor) and donor.origin=='native_capture' and donor.input_sha256==key and donor.context==context and donor.positions==positions and len(donor.values)==24
  self.mode=mode;self.input_sha=key;self.context=context;self.length=len(tokens);self.positions=positions;self.permutation=permutation;self.donor=donor;self.loops=loops;self.counts=[0]*24;self.events=[];self.flags=[];self.captured={};assert self.scorer is not None;self.scorer.invalidate()
 def verify(self):
  import torch
  assert self.counts==[self.loops]*24 and len(self.events)==self.loops*24
  expected=0 if self.mode=='off' else 24 if self.mode=='capture' else 48;assert len(self.flags)==expected
  if expected:assert bool(torch.stack(self.flags).all().item())
  if self.mode=='capture':assert set(self.captured)==set(range(24))
  self.verified=True;return dict(mode=self.mode,context=self.context,input_ids_sha256=self.input_sha,calls_per_layer=list(self.counts),events=list(self.events),source_positions=list(self.positions),permutation=list(self.permutation),other_rows_raw_exact=self.mode=='patch',native_donor_only=self.mode=='patch')
 def take_donor(self):
  assert self.verified and self.mode=='capture' and set(self.captured)==set(range(24))
  return NativeDonor(self.input_sha,self.context,self.positions,tuple(self.captured[i] for i in range(24)))
 def __exit__(self,*args):
  errors=[]
  for h in self.handles:
   try:h.remove()
   except BaseException as e:errors.append(repr(e))
  self.handles=[]
  if self.scorer is not None:
   try:self.scorer.invalidate()
   except BaseException as e:errors.append(repr(e))
  if [tuple(m._forward_hooks.items()) for m in self.modules]!=self.original:errors.append('Original V hooks not restored')
  self.cleanup=dict(handles_removed=True,original_hooks_restored=not errors,errors=errors)
  if errors and not(args and args[0] is not None):raise RuntimeError(errors)
