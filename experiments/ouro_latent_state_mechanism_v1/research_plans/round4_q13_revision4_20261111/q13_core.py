"""Q13 hybrid trajectory branch primitive; CPU draft, not approved for GPU."""
from contextlib import contextmanager
from dataclasses import dataclass
import torch,hashlib
if not __debug__:raise RuntimeError('optimized Python forbidden')
def bits_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def tensor_sha(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
@dataclass(frozen=True)
class Capture:
 trajectory:str
 input_ids_sha256:str
 layer_input:torch.Tensor
 branch:torch.Tensor
 branch_sha256:str
 input_sha256:str

def make_capture(trajectory,input_sha,record):
 if trajectory not in ('211','221','212','222') or not isinstance(input_sha,str) or len(input_sha)!=64:raise ValueError('identity')
 if record.get('visits')!=1 or not record.get('forward_complete'):raise ValueError('unsealed execution')
 x=record['input'].detach().clone();a=record['branch'].detach().clone()
 return Capture(trajectory,input_sha,x,a,tensor_sha(a),tensor_sha(x))

def matched(a,b,p,r):
 if a.input_ids_sha256!=b.input_ids_sha256 or (a.trajectory,b.trajectory) not in (('211','221'),('212','222')):raise ValueError('trajectory pairing')
 if not p or not r or set(p)&set(r):raise ValueError('regions')
 if not bits_equal(a.layer_input[:,p,:],b.layer_input[:,p,:]) or not bits_equal(a.layer_input[:,r,:],b.layer_input[:,r,:]):raise ValueError('P/R input mismatch')
 if a.input_sha256!=tensor_sha(a.layer_input) or b.input_sha256!=tensor_sha(b.layer_input) or a.branch_sha256!=tensor_sha(a.branch) or b.branch_sha256!=tensor_sha(b.branch):raise ValueError('donor mutation')
 return True

@contextmanager
def observe_branch(layer,positions,donor=None,receiver=None,prefix_positions=None):
 if layer.self_attn.layer_idx!=12:raise ValueError('layer index')
 if not positions or len(positions)!=len(set(positions)) or any(type(i)!=int or i<0 for i in positions):raise ValueError('positions')
 if donor is not None:
  if not isinstance(donor,Capture) or not isinstance(receiver,Capture) or receiver.input_sha256!=tensor_sha(receiver.layer_input) or donor.input_ids_sha256!=receiver.input_ids_sha256 or (donor.trajectory!=receiver.trajectory and (donor.trajectory,receiver.trajectory) not in (('211','221'),('221','211'),('212','222'),('222','212'))):raise ValueError('unmatched receiver')
  if not isinstance(donor,Capture) or donor.branch_sha256!=tensor_sha(donor.branch):raise ValueError('donor')
  if donor.trajectory!=receiver.trajectory:
   if not prefix_positions:raise ValueError('missing P region')
   lo,hi=(donor,receiver) if donor.trajectory in ('211','212') else (receiver,donor)
   matched(lo,hi,prefix_positions,positions)
  elif not bits_equal(donor.layer_input,receiver.layer_input) or donor.branch_sha256!=receiver.branch_sha256:raise ValueError('identity capture mismatch')
  private=donor.branch.detach().clone()
 else:private=None
 active=[];events=[];record={'visits':0,'forward_complete':False};handles=[]
 def pre(m,args,kw):
  u=kw.get('current_ut')
  if type(u)!=int or u!=len(events) or active or u>1:raise RuntimeError('loop sequence')
  x=args[0] if args else kw.get('hidden_states')
  if not torch.is_tensor(x) or x.ndim!=3 or x.shape[0]!=1 or max(positions)>=x.shape[1]:raise ValueError('layer input')
  active.append(u);events.append(u)
  if u==1:
   if receiver is not None and (receiver.input_sha256!=tensor_sha(receiver.layer_input) or not bits_equal(x,receiver.layer_input)):raise ValueError('receiver layer input mismatch')
   record['input']=x.detach().clone()
 def hook(m,args,out):
  if len(active)!=1:raise RuntimeError('branch outside layer')
  if active[0]!=1:return None
  if record['visits'] or not torch.is_tensor(out) or out.ndim!=3 or not torch.isfinite(out).all():raise ValueError('branch')
  record['visits']+=1;record['branch']=out.detach().clone()
  if private is None:return None
  if not isinstance(donor,Capture) or donor.branch_sha256!=tensor_sha(private) or private.shape!=out.shape or private.dtype!=out.dtype or private.device!=out.device:raise ValueError('donor identity')
  patched=out.clone();patched[:,positions,:]=private[:,positions,:]
  untouched=torch.ones(out.shape[1],dtype=torch.bool,device=out.device);untouched[positions]=False
  if not bits_equal(patched[:,untouched,:],out[:,untouched,:]):raise RuntimeError('outside R mutated')
  return patched
 def post(m,args,kw,out):
  if len(active)!=1:raise RuntimeError('bad scope')
  active.pop()
 try:
  handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True));handles.append(layer.input_layernorm_2.register_forward_hook(hook));handles.append(layer.register_forward_hook(post,with_kwargs=True))
  yield record
  if events!=[0,1] or active or record['visits']!=1:raise RuntimeError('incomplete')
  record['forward_complete']=True
 finally:
  for h in handles:h.remove()
