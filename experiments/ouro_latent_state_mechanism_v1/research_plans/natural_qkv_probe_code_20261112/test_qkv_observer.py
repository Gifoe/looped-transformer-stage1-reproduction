import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path.home()/'.codex/natural_branch_full_review_v4_20261111'))
sys.path.insert(0,str(Path.home()/'.codex/natural_l18_intervention_draft_20261111'))
import torch.nn.functional as F
from qkv_observer import observe_qkv,bits
from natural_l18_hook import intervene
torch.manual_seed(20);torch.set_num_threads(1)
class Attn(torch.nn.Module):
 def __init__(self,i):super().__init__();self.layer_idx=i;self.q=torch.nn.Linear(8,8,bias=False);self.k=torch.nn.Linear(8,8,bias=False);self.v=torch.nn.Linear(8,8,bias=False);self.o=torch.nn.Linear(8,8,bias=False)
 def qkv(self,x):
  b,l,h=x.shape;shape=(b,l,2,4);q=self.q(x).view(shape).transpose(1,2);k=self.k(x).view(shape).transpose(1,2);v=self.v(x).view(shape).transpose(1,2)
  theta=torch.arange(l,device=x.device,dtype=torch.float32)[None,None,:,None]*.13;cs=theta.cos().to(x.dtype);sn=theta.sin().to(x.dtype)
  def rot(t):a,b=t[...,:2],t[...,2:];return torch.cat((a*cs-b*sn,a*sn+b*cs),-1)
  return rot(q).contiguous(),rot(k).contiguous(),v.contiguous()
 def forward(self,hidden_states,current_ut=None,past_key_value=None,**kwargs):
  q,k,v=self.qkv(hidden_states);y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0);return self.o(y.transpose(1,2).reshape(hidden_states.shape))
class Layer(torch.nn.Module):
 def __init__(self,i):super().__init__();self.self_attn=Attn(i);self.input_layernorm=torch.nn.LayerNorm(8);self.input_layernorm_2=torch.nn.LayerNorm(8);self.post_attention_layernorm_2=torch.nn.LayerNorm(8);self.mlp=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.SiLU(),torch.nn.Linear(16,8))
 def forward(self,x,current_ut=None,use_cache=False,past_key_value=None):
  a=self.input_layernorm_2(self.self_attn(hidden_states=self.input_layernorm(x),current_ut=current_ut,past_key_value=past_key_value));y=x+a;return y+self.post_attention_layernorm_2(self.mlp(y))
class Core(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([Layer(i) for i in range(24)]);self.norm=torch.nn.LayerNorm(8)
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.model=Core()
 def forward(self,x,depth):
  for u in range(depth):
   for layer in self.model.layers:x=layer(x,current_ut=u,use_cache=False,past_key_value=None)
   x=self.model.norm(x)
  return x
@torch.inference_mode()
def manual(m,x,R,donor,q2,kv2):
 for i,l in enumerate(m.model.layers):
  h=l.input_layernorm(x);q,k,v=l.self_attn.qkv(h);base=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
  if i==17:
   qnew=q.clone()
   if q2:qnew[...,R,:]=donor['q'][...,R,:]
   hyb=F.scaled_dot_product_attention(qnew,donor['k'] if kv2 else k,donor['v'] if kv2 else v,is_causal=True,dropout_p=0.0);base=base.clone();base[...,R,:]=hyb[...,R,:]
  a=l.input_layernorm_2(l.self_attn.o(base.transpose(1,2).reshape(h.shape)));y=x+a;x=y+l.post_attention_layernorm_2(l.mlp(y))
 return m.model.norm(x)
@torch.inference_mode()
def main():
 m=Model().bfloat16().eval();x=torch.randn(1,7,8).bfloat16();R=4;old=F.scaled_dot_product_attention
 baseline=m(x,1)
 with observe_qkv(m,2,R) as donor:m(x,2)
 with intervene(m,2,R) as ca:m(x,2)
 with intervene(m,1,R) as cb:m(x,1)
 with intervene(m,1,R,cb['receiver_x'],cb['receiver_a'],ca['receiver_a']):t12=m(x,1)
 for q2,kv2 in ((False,False),(True,False),(False,True),(True,True)):
  with observe_qkv(m,1,R,donor,q2,kv2) as record:z=m(x,1)
  assert bits(z,manual(m,x,R,donor,q2,kv2))
  if not q2 and not kv2:assert bits(z,baseline)
  if q2 and kv2:assert bits(z,t12)
  assert bits(record['receiver_x'],cb['receiver_x'])
 assert F.scaled_dot_product_attention is old
 assert not m.model.layers[17]._forward_hooks and not m.model.layers[17]._forward_pre_hooks and not m.model.layers[17].self_attn._forward_hooks and not m.model.layers[17].self_attn._forward_pre_hooks
 try:
  invalid=dict(donor);invalid['k']=donor['k'][...,:-1,:]
  with observe_qkv(m,1,R,invalid,False,True):m(x,1)
 except ValueError:pass
 else:raise AssertionError('invalid K accepted')
 assert F.scaled_dot_product_attention is old
 try:
  invalid=dict(donor);invalid['kwargs']=dict(donor['kwargs']);invalid['kwargs']['is_causal']=False
  with observe_qkv(m,1,R,invalid,True,False):m(x,1)
 except ValueError:pass
 else:raise AssertionError('changed causal mask accepted')
 assert F.scaled_dot_product_attention is old
 try:
  invalid=dict(donor);invalid['v']=donor['v'].clone();invalid['v'][0,0,0,0]=float('nan')
  with observe_qkv(m,1,R,invalid,False,True):m(x,1)
 except ValueError:pass
 else:raise AssertionError('nonfinite donor accepted')
 assert F.scaled_dot_product_attention is old
 print('CPU_QKV_FOUR_CORNERS_FULL_SUFFIX_PASS')
if __name__=='__main__':main()
