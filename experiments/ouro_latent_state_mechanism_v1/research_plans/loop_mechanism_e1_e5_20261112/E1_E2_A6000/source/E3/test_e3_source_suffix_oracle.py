import sys,torch,torch.nn.functional as F
from pathlib import Path
sys.path.insert(0,str(Path.home()/'.codex/natural_qkv_probe_review_20261112'))
sys.path.insert(0,str(Path.home()/'.codex/natural_qkv_eight_corner_20261112'))
from test_qkv_observer import Model
from test_eight_gqa import GQAAttn
from eight_observer import observe_qkv,bits
from e1_injection_observer import inject_sdpa_R
from e3_source_control_v2 import build
@torch.inference_mode()
def manual(m,x,R,s):
 for i,l in enumerate(m.model.layers):
  h=l.input_layernorm(x);q,k,v=l.self_attn.qkv(h);base=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.,enable_gqa=True)
  if i==17:base=base.clone();base[...,R,:]=s
  a=l.input_layernorm_2(l.self_attn.o(base.transpose(1,2).reshape(h.shape)));y=x+a;x=y+l.post_attention_layernorm_2(l.mlp(y))
 return m.model.norm(x)
@torch.inference_mode()
def main():
 torch.manual_seed(102);m=Model()
 for i,l in enumerate(m.model.layers):l.self_attn=GQAAttn(i)
 m=m.bfloat16().eval();x=torch.randn(1,8,8).bfloat16();R=7;P=3
 with observe_qkv(m,1,R) as c1:m(x,1)
 with observe_qkv(m,2,R) as c2:m(x,2)
 for q in (1,2):
  qmix=c1['q'].clone()
  if q==2:qmix[...,R,:]=c2['q'][...,R,:]
  for name in ('ID','K_Q','V_Q','KV_Q','KV_P','K_Q_NORM','V_Q_NORM','V_Q_ROLL'):
   k,v,info=build(c1['k'],c2['k'],c1['v'],c2['v'],R,P,name);assert info['status']=='OK';hybrid=F.scaled_dot_product_attention(qmix,k,v,is_causal=True,dropout_p=0.,enable_gqa=True);s=hybrid[...,R,:]
   with inject_sdpa_R(m,R,s,capture_stages=False) as rec:y=m(x,1)
   assert bits(y,manual(m,x,R,s)) and rec['outside_R_unchanged']
   if name=='ID':
    with observe_qkv(m,1,R,c2,q==2,True,True) as ref:old=m(x,1)
    assert bits(y,old) and bits(s,ref['injected_sdpa_R'])
 print('E3_SOURCE_FULL_SHAPE_GQA_24_LAYER_CPU_ORACLE_PASS')
main()
