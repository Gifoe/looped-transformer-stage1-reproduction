"""Fixed loop-four source K/V capture/replay; no layer/head selection."""
class SourceReplay:
 def __init__(self,model):
  self.modules=[(kind,i,getattr(layer.self_attn,kind.lower()+'_proj')) for i,layer in enumerate(model.model.layers) for kind in ['K','V']]
  assert len(self.modules)==48
  self.original=[tuple(m._forward_hooks.items()) for _,_,m in self.modules];self.handles=[];self.scorer=None;self.verified=True;self.mode=None
 def bind(self,scorer):self.scorer=scorer
 def __enter__(self):
  try:
   for j,(kind,layer,m) in enumerate(self.modules):
    def hook(module,args,output,j=j,kind=kind,layer=layer):
     import torch
     assert self.mode is not None;self.verified=False;assert len(self.events)%48==j,'Projection event order mismatch';self.events.append((len(self.events)//48,layer,kind));self.counts[j]+=1;loop=self.counts[j];assert loop<=4
     assert output.ndim==3 and output.shape[0]==1 and output.shape[1]==self.length and output.dtype==torch.bfloat16
     if loop!=4:return None
     ix=torch.tensor(self.positions,device=output.device)
     if self.mode=='capture':
      self.captured[kind][layer]=output.index_select(1,ix).detach().clone();self.flags.append(torch.isfinite(self.captured[kind][layer]).all());return None
     if kind!=self.kind:return None
     donor=self.donor[layer];assert donor.shape==(1,4,output.shape[2]) and donor.dtype==output.dtype and donor.device==output.device
     source=donor.index_select(1,torch.tensor(self.permutation,device=output.device));result=output.clone();result.index_copy_(1,ix,source)
     self.flags.append((result.index_select(1,ix).contiguous().view(torch.uint8)==source.contiguous().view(torch.uint8)).all());mask=torch.ones(output.shape[1],dtype=torch.bool,device=output.device);mask[ix]=False;self.flags.append((result[:,mask,:].contiguous().view(torch.uint8)==output[:,mask,:].contiguous().view(torch.uint8)).all());return result
    self.handles.append(m.register_forward_hook(hook))
   return self
  except BaseException:
   self.__exit__(*__import__('sys').exc_info());raise
 def configure(self,mode,positions,length,kind=None,donor=None,permutation=(0,1,2,3)):
  assert self.verified,'Unverified previous forward cannot be overwritten'
  assert mode in ['capture','patch'] and len(positions)==len(set(positions))==4 and min(positions)>0 and max(positions)<length-1
  assert sorted(permutation)==[0,1,2,3]
  if mode=='patch':assert kind in ['K','V'] and len(donor)==24
  self.mode=mode;self.positions=tuple(positions);self.length=length;self.kind=kind;self.donor=donor;self.permutation=tuple(permutation);self.counts=[0]*48;self.events=[];self.flags=[];self.captured={k:{} for k in ['K','V']}
  assert self.scorer is not None;self.scorer.invalidate()
 def verify(self):
  import torch
  assert self.counts==[4]*48,self.counts
  assert len(self.events)==192
  if self.mode=='capture':assert all(set(self.captured[k])==set(range(24)) for k in ['K','V'])
  assert len(self.flags)==48 and bool(torch.stack(self.flags).all().item())
  self.verified=True
  return dict(mode=self.mode,kind=self.kind,calls_per_module=list(self.counts),source_positions=list(self.positions),permutation=list(self.permutation),captured_loop=4,events=list(self.events),other_rows_exact_verified=self.mode=='patch')
 def __exit__(self,*args):
  errors=[]
  for h in self.handles:
   try:h.remove()
   except BaseException as e:errors.append(repr(e))
  self.handles=[]
  if self.scorer is not None:
   try:self.scorer.invalidate()
   except BaseException as e:errors.append(repr(e))
  if [tuple(m._forward_hooks.items()) for _,_,m in self.modules]!=self.original:errors.append('Original hook dictionaries not restored')
  self.cleanup=dict(handles_removed=True,original_hooks_restored=not errors,errors=errors)
  if errors and not (args and args[0] is not None):raise RuntimeError(errors)
