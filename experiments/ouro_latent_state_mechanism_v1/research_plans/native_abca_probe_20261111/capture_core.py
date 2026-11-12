"""Non-entrypoint draft; must not be used until source, lock, resources and Astra review."""
import torch
from observer import observe
from integrity import validate_observations
@torch.inference_mode()
def capture_one(model,entry,prefix_length,expected_config):
 ids=entry['ids'];pos=entry['positions'];targets=entry['targets'];expected_scores=entry['scores']
 if len(pos)!=1 or not targets or len(expected_scores)!=len(targets):raise ValueError('scoring request')
 if model.training or model.config._attn_implementation!='sdpa' or next(model.parameters()).dtype!=torch.bfloat16:raise ValueError('model mode')
 previous=model.model.total_ut_steps;norm=[]
 def readout(m,a,out):norm.append(out[0,pos,:].detach().clone())
 handle=model.model.norm.register_forward_hook(readout)
 try:
  model.model.total_ut_steps=2;x=torch.tensor([ids],dtype=torch.long,device=model.lm_head.weight.device)
  with observe(model,prefix_length) as obs:
   output=model(x,labels=None,use_cache=False,exit_at_step=1)
  validate_observations(obs,prefix_length,len(ids),expected_config)
  if len(norm)!=2:raise ValueError('norm count')
  logits=model.lm_head(norm[-1].unsqueeze(0))[0];native=output.logits[0,pos,:]
  if logits.dtype!=torch.bfloat16 or not torch.isfinite(logits).all() or not torch.equal(logits,native):raise ValueError('head bridge')
  scores=[float(logits.float().log_softmax(-1).gather(-1,torch.tensor([[v]],device=logits.device)).sum()) for v in targets]
  if scores!=expected_scores:raise ValueError('natural K2 score bridge')
  if any(s['prefix'].dtype!=torch.bfloat16 for s in obs['states']):raise ValueError('state dtype')
  return obs,scores,logits.detach().cpu().clone()
 finally:
  handle.remove();model.model.total_ut_steps=previous
