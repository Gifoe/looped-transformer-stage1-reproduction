"""CPU staging only. Production source and real-model validation remain incomplete."""
import hashlib,torch
from q13_core import observe_branch,make_capture,matched,bits_equal,tensor_sha

def forward_capture(model,ids,trajectory,expected_cut,prefix,readout,run,source_guard,donor=None,receiver=None):
 if trajectory not in ('211','221','212','222'):raise ValueError('trajectory')
 if not torch.is_tensor(ids) or ids.ndim!=2 or ids.shape[0]!=1:raise ValueError('ids')
 ids_before=tensor_sha(ids);source_guard();root_events=[];cut_events=[];handles=[]
 def root(m,args,kwargs):
  actual=args[0] if args else kwargs.get('input_ids')
  if not torch.is_tensor(actual) or not bits_equal(actual,ids):raise ValueError('actual input differs')
  if kwargs.get('labels') is not None or kwargs.get('use_cache') is not False or kwargs.get('past_key_values') is not None:raise ValueError('forward mode')
  root_events.append(1)
  # Install after the controlled boundary hook so it sees the post-patch state.
  def cut(m,a,k,out):
   if k.get('current_ut')==1:
    if not bits_equal(out,expected_cut):raise ValueError('actual cutpoint differs')
    cut_events.append(tensor_sha(out))
  handles.append(model.model.layers[11].register_forward_hook(cut,with_kwargs=True))
 handles.append(model.register_forward_pre_hook(root,with_kwargs=True))
 try:
  with observe_branch(model.model.layers[12],readout,donor,receiver,prefix) as rec:
   logits=run()
  if not torch.is_tensor(logits) or not torch.isfinite(logits).all() or len(root_events)!=1 or len(cut_events)!=1:raise ValueError('incomplete forward')
  if tensor_sha(ids)!=ids_before:raise ValueError('input mutation')
  source_guard()
  return logits,make_capture(trajectory,ids_before,rec)
 finally:
  for h in handles:h.remove()

def run_matched_pair(model,ids,prefix,readout,run1,run2,cut1,cut2,trajectory1,trajectory2,source_guard):
 if (trajectory1,trajectory2) not in (('211','221'),('212','222')):raise ValueError('pair')
 z1,a=forward_capture(model,ids,trajectory1,cut1,prefix,readout,run1,source_guard)
 z2,b=forward_capture(model,ids,trajectory2,cut2,prefix,readout,run2,source_guard)
 matched(a,b,prefix,readout)
 i1,_=forward_capture(model,ids,trajectory1,cut1,prefix,readout,run1,source_guard,a,a)
 i2,_=forward_capture(model,ids,trajectory2,cut2,prefix,readout,run2,source_guard,b,b)
 if not bits_equal(z1,i1) or not bits_equal(z2,i2):raise ValueError('identity mismatch')
 u1,_=forward_capture(model,ids,trajectory1,cut1,prefix,readout,run1,source_guard,b,a)
 u2,_=forward_capture(model,ids,trajectory2,cut2,prefix,readout,run2,source_guard,a,b)
 if bits_equal(a.branch[:,readout,:],b.branch[:,readout,:]) and (not bits_equal(z1,u1) or not bits_equal(z2,u2)):raise ValueError('equal donor mismatch')
 return {'B1':z1,'B2':z2,'I1':i1,'I2':i2,'U1':u1,'U2':u2,'capture1':a,'capture2':b}
