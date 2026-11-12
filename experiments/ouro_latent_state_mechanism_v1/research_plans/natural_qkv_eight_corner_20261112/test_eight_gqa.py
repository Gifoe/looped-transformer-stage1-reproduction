"""Four-query-head/two-KV-head CPU fixture; not production GQA validation."""
import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path.home()/'.codex/natural_l18_intervention_draft_20261111'))
sys.path.insert(0,str(Path.home()/'.codex/natural_branch_full_review_v4_20261111'))
import torch.nn.functional as F
from eight_observer import observe_qkv,bits
from natural_l18_hook import intervene
from test_qkv_observer import Model,Attn
class GQAAttn(Attn):
 def __init__(self,i):
  torch.nn.Module.__init__(self);self.layer_idx=i;self.q=torch.nn.Linear(8,8,bias=False);self.k=torch.nn.Linear(8,4,bias=False);self.v=torch.nn.Linear(8,4,bias=False);self.o=torch.nn.Linear(8,8,bias=False)
 def qkv(self,x):
  b,l,h=x.shape;q=self.q(x).view(b,l,4,2).transpose(1,2);k=self.k(x).view(b,l,2,2).transpose(1,2);v=self.v(x).view(b,l,2,2).transpose(1,2);theta=torch.arange(l,dtype=torch.float32,device=x.device)[None,None,:,None]*.13;cs=theta.cos().to(x.dtype);sn=theta.sin().to(x.dtype)
  def rot(t):a,b=t[...,:1],t[...,1:];return torch.cat((a*cs-b*sn,a*sn+b*cs),-1)
  return rot(q).contiguous(),rot(k).contiguous(),v.contiguous()
 def forward(self,hidden_states,current_ut=None,past_key_value=None,**kwargs):
  q,k,v=self.qkv(hidden_states);y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.,enable_gqa=True);return self.o(y.transpose(1,2).reshape(hidden_states.shape))
@torch.inference_mode()
def manual(m,x,R,donor,q2,k2,v2):
 for i,l in enumerate(m.model.layers):
  h=l.input_layernorm(x);q,k,v=l.self_attn.qkv(h);kw={'is_causal':True,'dropout_p':0.,'enable_gqa':True};base=F.scaled_dot_product_attention(q,k,v,**kw)
  if i==17:
   qnew=q.clone()
   if q2:qnew[...,R,:]=donor['q'][...,R,:]
   hyb=F.scaled_dot_product_attention(qnew,donor['k'] if k2 else k,donor['v'] if v2 else v,**kw);base=base.clone();base[...,R,:]=hyb[...,R,:]
  a=l.input_layernorm_2(l.self_attn.o(base.transpose(1,2).reshape(h.shape)));y=x+a;x=y+l.post_attention_layernorm_2(l.mlp(y))
 return m.model.norm(x)
@torch.inference_mode()
def main():
 torch.set_num_threads(1);torch.manual_seed(71);m=Model()
 for i,l in enumerate(m.model.layers):l.self_attn=GQAAttn(i)
 m=m.bfloat16().eval();x=torch.randn(1,7,8).bfloat16();R=4;base=m(x,1)
 with observe_qkv(m,2,R) as donor:m(x,2)
 with intervene(m,2,R) as a2:m(x,2)
 with intervene(m,1,R) as a1:m(x,1)
 with intervene(m,1,R,a1['receiver_x'],a1['receiver_a'],a2['receiver_a']):t12=m(x,1)
 for q2,k2,v2 in ((a,b,c) for a in (False,True) for b in (False,True) for c in (False,True)):
  with observe_qkv(m,1,R,donor,q2,k2,v2) as record:z=m(x,1)
  assert record['q'].shape[1]==4 and record['k'].shape[1]==record['v'].shape[1]==2
  assert bits(z,manual(m,x,R,donor,q2,k2,v2))
  if not q2 and not k2 and not v2:assert bits(z,base)
  if q2 and k2 and v2:assert bits(z,t12)
 print('CPU_QKV_GQA_EIGHT_CORNERS_PASS')
if __name__=='__main__':main()
