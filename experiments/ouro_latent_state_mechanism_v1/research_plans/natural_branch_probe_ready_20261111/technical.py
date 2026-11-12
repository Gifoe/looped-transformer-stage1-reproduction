"""Natural model branch technical verification; CPU draft, no entrypoint."""
import torch
from state_capture import observe,bits

def score(z,targets):
 return [float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=z.device)).sum()) for t in targets]
@torch.inference_mode()
def verify(model,ids,R,targets,prior_states,prior_norms,prior_logits,prior_scores):
 if model.training or ids.ndim!=2 or ids.shape[0]!=1 or not 0<=R<ids.shape[1] or len(targets)<2:raise ValueError('input/model')
 core=model.model;old=core.total_ut_steps;nohook=[]
 try:
  for depth in (1,2):
   core.total_ut_steps=depth;y=model(ids,labels=None,use_cache=False,exit_at_step=depth-1);z=y.logits[0,R,:].unsqueeze(0)
   if not bits(z,prior_logits[depth-1].to(z.device)) or score(z,targets)!=prior_scores[depth-1]:raise ValueError('native baseline bridge')
   nohook.append(z.detach().clone())
  core.total_ut_steps=2
  with observe(model,R) as obs:y=model(ids,labels=None,use_cache=False,exit_at_step=1)
  if len(obs['attention'])!=48:raise ValueError('attention trace')
  expected_pos=torch.arange(ids.shape[1],dtype=torch.long).unsqueeze(0)
  import hashlib
  def digest(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  position_hash=digest(expected_pos);cache_hash=digest(expected_pos[0]);reference=None
  for row in obs['attention']:
   if row['position_ids']!={'shape':[1,ids.shape[1]],'dtype':'torch.int64','sha256':position_hash} or row['cache_position']!={'shape':[ids.shape[1]],'dtype':'torch.int64','sha256':cache_hash} or not row['past_is_none'] or row['use_cache'] is not False:raise ValueError('positions/cache trace')
   current={k:row[k] for k in ('attention_mask','position_ids','cache_position','rope','past_is_none','use_cache')}
   if reference is None:reference=current
   elif current!=reference:raise ValueError('attention parameters drift')
  zobs=[]
  for u in (0,1):
   for i in range(24):
    current=obs['states'][(u,i)]['z'];saved=prior_states[(u,i)].to(current.device)
    if not bits(current,saved):raise ValueError('original layer state bridge')
   if not bits(obs['norms'][u],prior_norms[u].to(obs['norms'][u].device)):raise ValueError('original norm bridge')
   z=model.lm_head(obs['norms'][u].unsqueeze(0))[0]
   if not bits(z,nohook[u]) or score(z,targets)!=prior_scores[u]:raise ValueError('observer/head bridge')
   zobs.append(z.detach().clone())
  if not bits(y.logits[0,R,:].unsqueeze(0),nohook[1]):raise ValueError('native output bridge')
  return obs,zobs
 finally:core.total_ut_steps=old
