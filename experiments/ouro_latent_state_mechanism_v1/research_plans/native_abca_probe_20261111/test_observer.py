import torch
import torch.nn.functional as F
from observer import observe
class Attention(torch.nn.Module):
 def __init__(self,i):super().__init__();self.layer_idx=i;self.swallow=False
 def forward(self,hidden_states,attention_mask,position_embeddings,past_key_value,use_cache):
  q=(hidden_states+position_embeddings[0]*0.001).unsqueeze(1)
  if self.swallow:
   try:F.scaled_dot_product_attention(q,q,q,attn_mask=torch.zeros(2,3))
   except RuntimeError:pass
  return F.scaled_dot_product_attention(q,q,q,attn_mask=attention_mask,dropout_p=0.,is_causal=attention_mask is None).squeeze(1)
class Layer(torch.nn.Module):
 def __init__(self,i):super().__init__();self.self_attn=Attention(i);self.mode=''
 def forward(self,x,**kw):
  if self.mode=='outside':
   q=x.unsqueeze(1);F.scaled_dot_product_attention(q,q,q,is_causal=True)
  vals={n:kw[n] for n in ('attention_mask','position_embeddings','past_key_value','use_cache')}
  if self.mode=='change_mask':vals['attention_mask']=torch.ones(x.shape[1],x.shape[1],dtype=torch.bool)
  if self.mode=='change_rope':vals['position_embeddings']=(torch.ones_like(x),torch.ones_like(x))
  if self.mode=='change_past':vals['past_key_value']='bad'
  if self.mode=='missing':del vals['past_key_value']
  return x+self.self_attn(hidden_states=x,**vals)*0.001
class Rope(torch.nn.Module):
 def forward(self,x,p):return x,x
class Core(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([Layer(i) for i in range(24)]);self.rotary_emb=Rope()
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.model=Core()
 def forward(self,x):
  p=torch.arange(x.shape[1]).unsqueeze(0);emb=self.model.rotary_emb(x,p)
  for u in range(2):
   for layer in self.model.layers:x=layer(x,current_ut=u,past_key_value=None,use_cache=False,attention_mask=None,position_embeddings=emb,cache_position=p)
  return x
m=Model();x=torch.randn(1,4,4);native=m(x);original=F.scaled_dot_product_attention
with observe(m,3) as rec:observed=m(x)
assert torch.equal(native,observed) and F.scaled_dot_product_attention is original
assert len(rec['sdpa'])==len(rec['attention'])==48 and all(z['is_causal'] is True and z['completed'] for z in rec['sdpa'])
assert all(z['prefix'].shape==(1,3,4) for z in rec['states'])
for mode in ('change_mask','change_rope'):
 m.model.layers[11].mode=mode
 with observe(m,3) as rec:m(x)
 actual=rec['attention'][11]
 if mode=='change_mask':assert torch.is_tensor(actual['attention_mask']) and rec['sdpa'][11]['is_causal'] is False
 else:assert torch.equal(actual['position_embeddings'][0],torch.ones_like(x))
for mode in ('outside','change_past','missing'):
 m.model.layers[11].mode=mode
 try:
  with observe(m,3):m(x)
 except RuntimeError:pass
 else:raise AssertionError(mode)
 assert F.scaled_dot_product_attention is original and not m.model.layers[11]._forward_hooks
m.model.layers[11].mode='';m.model.layers[11].self_attn.swallow=True
try:
 with observe(m,3):m(x)
except RuntimeError:pass
else:raise AssertionError('swallowed failed SDPA accepted')
try:
 with observe(m,3):pass
except RuntimeError:pass
else:raise AssertionError('missing forward accepted')
assert F.scaled_dot_product_attention is original
print('TOY_PASS 48 paired self-attention events, changed actual args, external/failed/missing rejection; NOT GPU')
