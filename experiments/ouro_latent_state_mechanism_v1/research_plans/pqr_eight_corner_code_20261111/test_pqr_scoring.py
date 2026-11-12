"""CPU toy end-to-end scoring path; not a GPU gate."""
import sys,torch
from torch import nn
from types import SimpleNamespace
sys.path.insert(0,'/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1')
from round2_full_sequence import full_boundary
from pqr_core import run_question
from test_pqr_core import Core
class Toy(nn.Module):
 def __init__(self):
  super().__init__();self.model=Core();self.model.norm=nn.Identity();self.lm_head=nn.Linear(2,7,bias=False,dtype=torch.bfloat16);self.embed=nn.Embedding(10,2,dtype=torch.bfloat16)
 def forward(self,ids,labels=None,use_cache=False,exit_at_step=None):
  h=self.embed(ids)
  for u in range(self.model.total_ut_steps):
   for l in self.model.layers:h=l(h,current_ut=u)
   h=self.model.norm(h)
  return SimpleNamespace(logits=self.lm_head(h))
torch.set_num_threads(1);torch.manual_seed(8);m=Toy().eval();x=torch.tensor([[1,2,3,4,5]]);pos=[4];targets=[1,2,3];regions={'P':[0,1],'Q':[2,3],'R':[4]}
def f(depth,**kw):
 norms=[];hook=m.model.norm.register_forward_hook(lambda mod,args,out:norms.append(out[0,pos,:].detach().clone()))
 try:
  with full_boundary(m,depth,**kw) as states:m(x,labels=None,use_cache=False,exit_at_step=depth-1)
 finally:hook.remove()
 z=m.lm_head(norms[-1].unsqueeze(0))[0];scores=[float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]])).sum()) for t in targets];return states,scores,z.detach()
s1,b1,z1=f(1);s2,b2,z2=f(2);k1=s1[0];k2=s2[1];gold=0;wrong=max((i for i in range(3) if i!=gold),key=lambda i:(b1[i],-i));
def pack(v):
 pred=max(range(len(v)),key=lambda i:(v[i],-i));return {'scores':v,'prediction':pred,'top_tie':sum(z==v[pred] for z in v)>1,'gold_vs_fixed_wrong_gap':v[gold]-v[wrong]}
old={'baseline_K1':pack(b1),'baseline_K2':pack(b2),'gold':gold,'k1_highest_wrong':wrong,'conditions':{}};oldz={'baseline_K1':z1,'baseline_K2':z2}
for c in ('K1C_K1R','K1C_K2R','K2C_K1R','K2C_K2R'):
 _,v,z=f(2,condition=c,k1=k1,k2_reference=k2,readout=pos);old['conditions'][c]=pack(v);oldz['core_'+c]=z
records=[{'candidate':i,'scores':[b1[i],b2[i]],'encoding':{'input_ids':x[0].tolist(),'predecessor_positions':pos,'target_ids':[t]}} for i,t in enumerate(targets)]
result,logits=run_question(m,records,gold,regions,old,oldz);assert set(result['corners'])=={'111','112','121','122','211','212','221','222'} and set(logits)=={'corner_121','corner_122','corner_211','corner_212'}
oldz['core_K1C_K2R']=oldz['core_K1C_K2R'].clone();oldz['core_K1C_K2R'][0,0]+=1
try:run_question(m,records,gold,regions,old,oldz)
except RuntimeError:pass
else:raise AssertionError('old logits mismatch accepted')
print('PQR_SCORING_SYNTHETIC_CPU_PASS')
