import sys,torch,torch.nn.functional as F
from pathlib import Path
sys.path.insert(0,str(Path.home()/'.codex/natural_qkv_probe_review_20261112'))
sys.path.insert(0,str(Path.home()/'.codex/natural_qkv_eight_corner_20261112'))
from test_qkv_observer import Model
from eight_observer import observe_qkv,bits
from e1_injection_observer import inject_sdpa_R
@torch.inference_mode()
def manual(m,x,R,value):
 for i,l in enumerate(m.model.layers):
  h=l.input_layernorm(x);q,k,v=l.self_attn.qkv(h);base=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
  if i==17:base=base.clone();base[...,R,:]=value
  a=l.input_layernorm_2(l.self_attn.o(base.transpose(1,2).reshape(h.shape)));y=x+a;x=y+l.post_attention_layernorm_2(l.mlp(y))
 return m.model.norm(x)
@torch.inference_mode()
def main():
 torch.manual_seed(91);m=Model().bfloat16().eval();x=torch.randn(1,7,8).bfloat16();R=4;old=F.scaled_dot_product_attention;baseline=m(x,1)
 with observe_qkv(m,1,R) as q:m(x,1)
 value=q['baseline_sdpa_R']
 with inject_sdpa_R(m,R,value,capture_stages=False) as rec:y=m(x,1)
 assert bits(y,baseline) and rec['outside_R_unchanged']
 changed=value+torch.full_like(value,.125)
 with inject_sdpa_R(m,R,changed,capture_stages=False):y=m(x,1)
 assert bits(y,manual(m,x,R,changed)) and F.scaled_dot_product_attention is old
 try:
  with inject_sdpa_R(m,R,changed,expected_x=torch.full((1,8),123,dtype=torch.bfloat16),capture_stages=False):m(x,1)
 except ValueError:pass
 else:raise AssertionError('wrong receiver allowed')
 assert F.scaled_dot_product_attention is old
 print('E1_R_PATCH_24_LAYER_CPU_ORACLE_PASS')
main()
