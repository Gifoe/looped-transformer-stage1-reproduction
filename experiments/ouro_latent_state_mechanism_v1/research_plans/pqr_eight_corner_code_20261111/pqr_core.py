"""Pre-review CPU/GPU core DRAFT; no scientific or production approval."""
from contextlib import contextmanager
import torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
CORNERS=('111','112','121','122','211','212','221','222'); OLD={'111':'K1C_K1R','112':'K1C_K2R','221':'K2C_K1R','222':'K2C_K2R'}; NEW=('121','122','211','212')
def compose(k1,k2,regions,corner):
 if corner not in CORNERS or set(regions)!={'P','Q','R'}:raise ValueError('condition/regions')
 if not torch.is_tensor(k1) or not torch.is_tensor(k2) or k1.ndim!=3 or k1.shape!=k2.shape or k1.shape[0]!=1 or k1.dtype!=k2.dtype or k1.device!=k2.device:raise ValueError('state')
 if not torch.isfinite(k1).all() or not torch.isfinite(k2).all():raise ValueError('nonfinite state')
 groups=[]
 for k in ('P','Q','R'):
  v=list(regions[k]);
  if not v or len(v)!=len(set(v)) or any(type(i)!=int or not 0<=i<k1.shape[1] for i in v):raise ValueError('empty/invalid region')
  groups.append(set(v))
 if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)) or set.union(*groups)!=set(range(k1.shape[1])):raise ValueError('region coverage')
 out=k2.clone()
 for bit,k in zip(corner,('P','Q','R')):
  if bit=='1':out[:,regions[k],:]=k1[:,regions[k],:]
 for bit,k in zip(corner,('P','Q','R')):
  if not torch.equal(out[:,regions[k],:],(k1 if bit=='1' else k2)[:,regions[k],:]):raise RuntimeError('composition mismatch')
 return out
@contextmanager
def pqr_boundary(model,k1,k2,regions,corner):
 core=model.model
 if len(core.layers)!=24 or core.layers[11].self_attn.layer_idx!=11:raise ValueError('physical layer')
 old=core.total_ut_steps;layer=core.layers[11];handles=[];events=[];active=[]
 def rootpre(m,args,kw):
  if kw.get('labels') is not None or kw.get('use_cache') is not False or kw.get('past_key_values') is not None:raise RuntimeError('input mode')
 def pre(m,args,kw):
  u=kw.get('current_ut')
  if type(u)!=int or u!=len(events) or u>=2 or active:raise RuntimeError('loop order')
  events.append(u);active.append(u)
 def post(m,args,kw,out):
  if len(active)!=1 or not torch.is_tensor(out):raise RuntimeError('bad output')
  u=active.pop()
  if u==1:
   if not torch.equal(out,k2):raise RuntimeError('native K2 state drift')
   return compose(k1,out,regions,corner)
 try:
  core.total_ut_steps=2;handles.append(model.register_forward_pre_hook(rootpre,with_kwargs=True));handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True));handles.append(layer.register_forward_hook(post,with_kwargs=True));yield
  if events!=[0,1] or active:raise RuntimeError('event sequence')
 finally:
  for h in handles:h.remove()
  core.total_ut_steps=old
@torch.inference_mode()
def run_question(model,records,gold,regions,old_result,old_logits):
 """True computation; only call after full runner provenance/gate acceptance."""
 from round2_full_sequence import full_boundary
 records=sorted(records,key=lambda x:x['candidate']);assert len(records)>=2 and [r['candidate'] for r in records]==list(range(len(records)))
 ids=records[0]['encoding']['input_ids'];pos=records[0]['encoding']['predecessor_positions'];assert len(pos)==1 and pos==regions['R'] and 0<=gold<len(records)
 assert all(r['encoding']['input_ids']==ids and r['encoding']['predecessor_positions']==pos and len(r['encoding']['target_ids'])==1 for r in records)
 targets=[r['encoding']['target_ids'][0] for r in records];x=torch.tensor([ids],dtype=torch.long,device=model.lm_head.weight.device)
 def forward(depth,corner=None,k1=None,k2=None,layer_index=11,legacy=None):
  norm=[]
  def collect(m,a,out):norm.append(out[0,pos,:].detach().clone())
  h=model.model.norm.register_forward_hook(collect)
  try:
   if corner is None:
    with full_boundary(model,depth,layer_index=layer_index,**(legacy or {})) as states:y=model(x,labels=None,use_cache=False,exit_at_step=depth-1)
   else:
    states=None
    with pqr_boundary(model,k1,k2,regions,corner):y=model(x,labels=None,use_cache=False,exit_at_step=1)
  finally:h.remove()
  if len(norm)!=depth:raise RuntimeError('norm count')
  z=model.lm_head(norm[-1].unsqueeze(0))[0];native=y.logits[0,pos,:].detach().clone();del y
  if z.dtype!=torch.bfloat16 or not torch.isfinite(z).all() or not torch.equal(z,native):raise RuntimeError('projection bridge')
  scores=[float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=z.device)).sum()) for t in targets]
  return states,scores,z.detach().clone()
 s1,b1,z1=forward(1);s2,b2,z2=forward(2);k1=s1[0];k2=s2[1]
 assert b1==[r['scores'][0] for r in records] and b2==[r['scores'][1] for r in records]
 assert old_result['baseline_K1']['scores']==b1 and old_result['baseline_K2']['scores']==b2
 wrong=max((i for i in range(len(targets)) if i!=gold),key=lambda i:(b1[i],-i));assert wrong==old_result['k1_highest_wrong'] and old_result['gold']==gold
 results={};stored={}
 for corner in CORNERS:
  _,scores,z=forward(2,corner,k1,k2);prediction=max(range(len(scores)),key=lambda i:(scores[i],-i));entry={'scores':scores,'prediction':prediction,'top_tie':sum(x==scores[prediction] for x in scores)>1,'gold_vs_fixed_wrong_gap':scores[gold]-scores[wrong]}
  if corner in OLD:
   old=OLD[corner];expected=old_result['conditions'][old];oldz=old_logits['core_'+old].to(z.device)
   if entry!=expected or not torch.equal(z,oldz):raise RuntimeError('old corner bridge '+corner)
  results[corner]=entry
  if corner in NEW:stored['corner_'+corner]=z.cpu().contiguous()
 if not torch.equal(z1,old_logits['baseline_K1'].to(z1.device)) or not torch.equal(z2,old_logits['baseline_K2'].to(z2.device)):raise RuntimeError('native logits bridge')
 l1,_,_=forward(1,layer_index=23);l2,_,_=forward(2,layer_index=23);_,neg,zn=forward(2,layer_index=23,legacy={'condition':'K1C_K2R','k1':l1[0],'k2_reference':l2[1],'readout':pos})
 if neg!=b2 or not torch.equal(zn,z2):raise RuntimeError('last-layer context negative control')
 return {'gold':gold,'fixed_wrong':wrong,'target_ids':targets,'regions':regions,'corners':results,'anchors':{'original_four':'exact','native_K1_K2':'exact','last_layer_C':'exact'}},stored
