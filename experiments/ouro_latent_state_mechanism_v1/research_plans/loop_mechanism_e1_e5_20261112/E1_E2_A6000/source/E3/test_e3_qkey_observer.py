import torch
from test_e3_producer_four_corner_cpu import Model,manual,bits
from e3_qkey_producer_observer import producer_observer
@torch.inference_mode()
def test():
 torch.manual_seed(143);torch.set_num_threads(1);m=Model().bfloat16().eval();x=torch.randn(1,9,8).bfloat16();R=8;P=4
 for cut in (0,8,16,17):
  with producer_observer(m,cut,R,P) as clean:y0=m(x,2)
  h=clean['loop1_cut_hidden'];k0=clean['k']
  with producer_observer(m,cut,R,P,hidden_donor=h) as pert:y1=m(x,2)
  k1=pert['k']
  if cut==17:
   t=x.clone()
   for layer in m.model.layers[:17]:
    a=layer(t,current_ut=0,use_cache=False,past_key_value=None);t=a
   _,kfirst,_=m.model.layers[17].self_attn.qkv(m.model.layers[17].input_layernorm(t))
   assert bits(k1[...,P:R,:],kfirst[...,P:R,:]) and bits(k1[...,:P,:],k0[...,:P,:]) and bits(k1[...,R:,:],k0[...,R:,:]) and bits(pert['q'][...,R,:],clean['q'][...,R,:])
  for h1,k in ((None,None),(None,k1),(h,None),(h,k0)):
   with producer_observer(m,cut,R,P,hidden_donor=h1,key_donor=k) as r:z=m(x,2)
   assert bits(z,manual(m,x,cut,h,R,P,h1 is not None,k))
  assert bits(y0,manual(m,x,cut,h,R,P,False,None)) and bits(y1,manual(m,x,cut,h,R,P,True,None))
 print('PRODUCTION_OBSERVER_FOUR_CORNERS_CPU_PASS')
test()
