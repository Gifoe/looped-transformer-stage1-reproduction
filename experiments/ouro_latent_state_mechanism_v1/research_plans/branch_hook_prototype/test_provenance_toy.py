import json,torch
from torch import nn
from test_contract import Layer,Norm
from provenance_toy import capture,verify,tensor_identity,CapturedDonor
torch.set_num_threads(1);torch.manual_seed(31)
class Model(nn.Module):
 def __init__(self):super().__init__();self.layers=nn.ModuleList([Layer(i,4) for i in range(24)]);self.embedding=nn.Embedding(100,4);self.norm=Norm(4)
 def forward(self,ids,mask,pos,depth):
  x=self.embedding(ids)
  for u in range(depth):
   for l in self.layers:x=l(x,current_ut=u)
   x=self.norm(x)
  return x
m=Model();ids=torch.tensor([[1,2,3,4,5]]);mask=torch.ones_like(ids);pos=torch.arange(5).reshape(1,5)
manifest={'question_uid':'toy-q1','metadata_sha256':'a'*64,'ids':tensor_identity(ids),'mask':tensor_identity(mask),'positions':tensor_identity(pos),'targets':[3,4], 'readouts':[4]}
d,out=capture(m,ids,mask,pos,manifest,1,11,0,'attention');assert torch.equal(verify(d,m,ids,mask,pos,manifest,11,'attention',(1,0),[]),d.tensor)
def rejects(f):
 try:f()
 except (ValueError,TypeError,NotImplementedError):return
 raise AssertionError('accepted invalid case')
bad=dict(manifest);bad['question_uid']='toy-q2';rejects(lambda:verify(d,m,ids,mask,pos,bad,11,'attention',(1,0),[]))
new=ids.clone();new[0,0]=9;rejects(lambda:verify(d,m,new,mask,pos,manifest,11,'attention',(1,0),[]))
rejects(lambda:verify(d,m,ids,mask,pos,manifest,10,'attention',(1,0),[]));rejects(lambda:verify(d,m,ids,mask,pos,manifest,11,'mlp',(1,0),[]));rejects(lambda:verify(d,m,ids,mask,pos,manifest,11,'attention',(2,0),[]));rejects(lambda:verify(d,m,ids,mask,pos,manifest,11,'attention',(1,0),['prior_patch'],True))
raw=json.loads(d.receipt);raw['forward_complete']=False;rejects(lambda:verify(CapturedDonor(d.seal,json.dumps(raw).encode(),d.tensor),m,ids,mask,pos,manifest,11,'attention',(1,0),[]))
modified=d.tensor.clone();modified[0,0,0]+=1;rejects(lambda:verify(CapturedDonor(d.seal,d.receipt,modified),m,ids,mask,pos,manifest,11,'attention',(1,0),[]))
handle=m.layers[0].register_forward_hook(lambda *a:None)
try:rejects(lambda:capture(m,ids,mask,pos,manifest,1,11,0,'attention'))
finally:handle.remove()
d2,_=capture(m,ids,mask,pos,manifest,2,11,0,'attention');assert json.loads(d2.receipt)['source_kind']=='K2_run_first_pass';rejects(lambda:verify(d2,m,ids,mask,pos,manifest,11,'attention',(1,0),[]))
from provenance_toy import patch_forward,_SEALS
patched=patch_forward(d,m,ids,mask,pos,manifest,2,11,1,'attention','R',(1,0));assert torch.isfinite(patched).all()
rejects(lambda:patch_forward(d,m,ids,mask,pos,manifest,2,11,1,'attention','INVALID',(1,0)))
class Mutating(Model):
 def __init__(self,mode):super().__init__();self.mode=mode
 def forward(self,ids,mask,pos,depth):
  if self.mode=='ids':ids[0,0]=7
  out=super().forward(ids,mask,pos,depth)
  if self.mode=='weight':
   with torch.no_grad():self.layers[-1].mlp.c.weight[0,0]+=1
  if self.mode=='nan':out=out*float('nan')
  return out
for mode in ('ids','weight','nan'):
 model=Mutating(mode); local=ids.clone();count=len(_SEALS)
 rejects(lambda:capture(model,local,mask,pos,manifest,1,11,0,'attention'))
 assert len(_SEALS)==count
# Independent complete suffix oracle on receiver loop 1 input.
seen={}
def read_input(module,args,kwargs):
 if kwargs['current_ut']==1:seen['x']=args[0].detach().clone()
h=m.layers[11].register_forward_pre_hook(read_input,with_kwargs=True)
try:m(ids,mask,pos,2)
finally:h.remove()
x2=seen['x'];layer=m.layers[11];a=layer.input_layernorm_2(layer.self_attn(layer.input_layernorm(x2)));a=a.clone();a[:,4,:]=d.tensor[:,4,:];y=x2+a;oracle=y+layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(y)))
for layer in m.layers[12:]:oracle=layer(oracle,current_ut=1)
oracle=m.norm(oracle);assert torch.equal(patched,oracle)
oldidx=m.layers[11].self_attn.layer_idx;m.layers[11].self_attn.layer_idx=10
try:rejects(lambda:patch_forward(d,m,ids,mask,pos,manifest,2,11,1,'attention','R',(1,0)))
finally:m.layers[11].self_attn.layer_idx=oldidx
oldforward=m.forward
def mutate_donor(*args):
 d.tensor[0,4,0]+=100
 return oldforward(*args)
m.forward=mutate_donor
try:stable=patch_forward(d,m,ids,mask,pos,manifest,2,11,1,'attention','R',(1,0))
finally:m.forward=oldforward
assert torch.equal(stable,oracle)
print('TOY_PARTIAL_RECEIVER_ORACLE_PASS in-memory only')
