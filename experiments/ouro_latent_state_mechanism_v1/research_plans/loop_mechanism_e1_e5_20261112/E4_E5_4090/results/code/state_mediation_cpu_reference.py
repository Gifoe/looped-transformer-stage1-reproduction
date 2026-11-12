"""CPU-only causal/hook check; this is not an Ouro experimental result."""
import json,math,torch
from torch import nn
from torch.nn import functional as F
from state_mediation_components import input_r_block

class ToyLayer(nn.Module):
 def forward(self,x,*,current_ut,use_cache,past_key_value):
  return F.scaled_dot_product_attention(x[:,None],x[:,None],x[:,None],is_causal=True).squeeze(1)
class Toy(nn.Module):
 def __init__(self):
  super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList(ToyLayer() for unused in range(24))
 def forward(self,x):
  for u in range(2):x=self.model.layers[17](x,current_ut=u,use_cache=False,past_key_value=None)
  return x
torch.manual_seed(184);model=Toy().eval();x=torch.randn(1,5,4,dtype=torch.float64);R=4;old=F.scaled_dot_product_attention
with input_r_block(model,2,R) as native:model(x)
with input_r_block(model,2,R,x[:,R,:].clone()) as blocked:model(x)
with input_r_block(model,2,R,native['original_x_R']) as identical:model(x)
assert F.scaled_dot_product_attention is old
for name in ('x','Q','K','V'):
 assert native['non_R_'+name+'_sha256']==blocked['non_R_'+name+'_sha256']
assert native['non_R_sdpa_sha256']==blocked['non_R_sdpa_sha256']
assert torch.equal(native['actual_sdpa_R'],identical['actual_sdpa_R'])
first=model.model.layers[17](x,current_ut=0,use_cache=False,past_key_value=None);changed=first.clone();changed[:,R,:]=x[:,R,:]
reference=(torch.softmax(changed[:,R:R+1,:]@changed.transpose(-2,-1)/math.sqrt(4),dim=-1)@changed)[:,0,:].unsqueeze(1)
assert torch.allclose(reference,blocked['actual_sdpa_R'],atol=1e-12,rtol=1e-12)
assert not torch.equal(blocked['actual_sdpa_R'],native['actual_sdpa_R'])
assert not torch.equal(blocked['actual_sdpa_R'],first[:,R,:].unsqueeze(1))
print(json.dumps({'status':'CPU_CAUSAL_R_BLOCK_REFERENCE_PASSED','scope':'CPU_TOY_NOT_OURO_MODEL_EXPERIMENT','non_R_x_Q_K_V_SDPA_hashes_unchanged':True,'identity_exact':True,'actual_blocked_attention_matches_independent_FP64_softmax':True,'A_U_distinct_from_native_A2_and_natural_A1_in_this_toy':True}))
