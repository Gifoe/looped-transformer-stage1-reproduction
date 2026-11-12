import torch
from types import SimpleNamespace
from test_observer import Core
from capture_core import capture_one
class Mock(torch.nn.Module):
 def __init__(self):
  super().__init__();self.model=Core();self.model.total_ut_steps=4;self.model.norm=torch.nn.Identity();self.embed=torch.nn.Embedding(16,4).bfloat16();self.lm_head=torch.nn.Linear(4,16,bias=False).bfloat16();self.config=SimpleNamespace(_attn_implementation='sdpa')
 def forward(self,ids,labels=None,use_cache=None,exit_at_step=None):
  assert labels is None and use_cache is False and exit_at_step==1 and self.model.total_ut_steps==2
  x=self.embed(ids);p=torch.arange(x.shape[1]).unsqueeze(0);emb=self.model.rotary_emb(x,p)
  for u in range(2):
   for layer in self.model.layers:x=layer(x,current_ut=u,past_key_value=None,use_cache=False,attention_mask=None,position_embeddings=emb,cache_position=p)
   x=self.model.norm(x)
  return SimpleNamespace(logits=self.lm_head(x))
m=Mock().eval();entry={'ids':[1,2,3,4],'positions':[3],'targets':[5,6]};m.model.total_ut_steps=2
with torch.inference_mode():
 logits=m(torch.tensor([[1,2,3,4]]),labels=None,use_cache=False,exit_at_step=1).logits[0,[3],:]
 entry['scores']=[float(logits.float().log_softmax(-1).gather(-1,torch.tensor([[v]])).sum()) for v in entry['targets']]
m.model.total_ut_steps=4
cfg={'head_dim':4,'hidden_size':4,'q_heads':1,'kv_heads':1,'dtype':'torch.bfloat16','device':'cpu'}
obs,scores,z=capture_one(m,entry,3,cfg)
assert scores==entry['scores'] and len(obs['states'])==2 and torch.equal(z,logits.cpu()) and m.model.total_ut_steps==4 and not m.model.norm._forward_hooks
bad=dict(entry);bad['scores']=[999.,999.]
try:capture_one(m,bad,3,cfg)
except ValueError as e:assert 'score' in str(e)
else:raise AssertionError('score mismatch accepted')
assert m.model.total_ut_steps==4 and not m.model.norm._forward_hooks
print('CAPTURE_MOCK_PASS: real control flow, config, score bridge, cleanup; NOT GPU')
