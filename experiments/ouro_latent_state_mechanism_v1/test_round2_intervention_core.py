"""CPU synthetic intervention audit: structural only, not native model gate."""
import types,torch
from torch import nn
from round2_full_sequence import full_boundary
from round2_intervention_core import run_question,CORE,NORM
class Layer(nn.Module):
 def __init__(self,i):super().__init__();self.self_attn=types.SimpleNamespace(layer_idx=i)
 def forward(self,x,current_ut=None):return x+(x.cumsum(dim=1)*0.015625).to(x.dtype)
class Core(nn.Module):
 def __init__(self):super().__init__();self.layers=nn.ModuleList([Layer(i) for i in range(24)]);self.norm=nn.Identity();self.total_ut_steps=4
class Toy(nn.Module):
 def __init__(self):
  super().__init__();self.model=Core();self.emb=nn.Embedding(16,4,dtype=torch.bfloat16);self.lm_head=nn.Linear(4,16,bias=False,dtype=torch.bfloat16)
  torch.manual_seed(27);nn.init.uniform_(self.emb.weight,-.25,.25);nn.init.uniform_(self.lm_head.weight,-.25,.25)
 def forward(self,x,labels=None,use_cache=None,past_key_values=None,exit_at_step=None):
  h=self.emb(x)
  for k in range(self.model.total_ut_steps):
   for layer in self.model.layers:h=layer(h,current_ut=k)
   h=self.model.norm(h)
  return types.SimpleNamespace(logits=self.lm_head(h))
m=Toy().eval();ids=[1,2,3,4];pos=[3];targets=[2,3,4];record=[{'candidate':i,'encoding':{'input_ids':ids,'predecessor_positions':pos,'target_ids':[t]},'scores':[0,0]} for i,t in enumerate(targets)];x=torch.tensor([ids])
for depth in (1,2):
 with torch.inference_mode():
  with full_boundary(m,depth):out=m(x,labels=None,use_cache=False,exit_at_step=depth-1)
  z=out.logits[0,pos,:].float().log_softmax(-1)
  for i,t in enumerate(targets):record[i]['scores'][depth-1]=float(z[0,t])
res,logits=run_question(m,record,1);assert set(logits)=={'baseline_K1','baseline_K2','negative_last_C'}|{'core_'+x for x in CORE}|{'norm_'+x for x in NORM};assert all(v.dtype==torch.bfloat16 and v.shape==(1,16) for v in logits.values());assert set(res['conditions'])==set(CORE) and set(res['norm_conditions'])==set(NORM);assert res['controls']=={'K2_identity':'exact','K1_full_anchor':'exact','last_layer_C_only':'exact','rescale_K2_to_K2':'exact'};assert res['readout_positions']==pos and res['context_positions_count']==3 and len(res['norms']['K1DIR_K2NORM']['actual_bf16_norm'][0])==4
assert m.model.total_ut_steps==4 and all(not q._forward_hooks and not q._forward_pre_hooks for q in m.model.layers)
print('PASS synthetic complete question, 4 core + 4 norm controls, exact anchors/negative control, norm measurement and hook cleanup; NOT GPU acceptance')
