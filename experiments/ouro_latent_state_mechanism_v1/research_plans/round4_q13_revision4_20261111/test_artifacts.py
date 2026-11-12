import tempfile,torch,json,hashlib
from pathlib import Path
from artifacts import save,replay,canonical
from q13_core import Capture,tensor_sha
source='f'*64;identity={'id':'test','input_ids_tensor_sha256':source};targets=[1,2];prefix=[0];readout=[4];guard=lambda:None
x=torch.zeros((1,5,3),dtype=torch.bfloat16)
def build(equal=False):
 results={};prior={}
 for r in ('1','2'):
  a=torch.ones((1,5,3),dtype=torch.bfloat16)*int(r);b=a.clone() if equal else a+1
  a1=Capture('21'+r,source,x.clone(),a,tensor_sha(a),tensor_sha(x));a2=Capture('22'+r,source,x.clone(),b,tensor_sha(b),tensor_sha(x))
  z1=torch.tensor([[1.,2.,3.,4.]],dtype=torch.bfloat16)+int(r);z2=z1+2
  results[r]={'B1':z1,'B2':z2,'I1':z1.clone(),'I2':z2.clone(),'U1':z1.clone() if equal else z1+3,'U2':z2.clone() if equal else z2+4,'capture1':a1,'capture2':a2};prior['21'+r]=z1;prior['22'+r]=z2
 return results,prior
for equal in (False,True):
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'q';results,prior=build(equal);save(p,identity,results,targets,prefix,readout,4,3,prior,guard)
  assert replay(p,identity,targets,prefix,readout,4,3,prior,'cpu',guard)['status']=='ZERO_SCORE_DIFFERENCE'
  try:save(p,identity,results,targets,prefix,readout,4,3,prior,guard)
  except FileExistsError:pass
  else:raise AssertionError('overwrite')
  f=p/'values.safetensors';original=f.read_bytes();v=bytearray(original);v[-1]^=1;f.write_bytes(v)
  try:replay(p,identity,targets,prefix,readout,4,3,prior,'cpu',guard)
  except ValueError:pass
  else:raise AssertionError('corrupt accepted')
  f.write_bytes(original);m=p/'metadata.json';j=json.loads(m.read_text());j['payload']['captures']['a1r1']['trajectory']='221';j['payload_sha256']=hashlib.sha256(canonical(j['payload'])).hexdigest();m.write_bytes(canonical(j))
  try:replay(p,identity,targets,prefix,readout,4,3,prior,'cpu',guard)
  except ValueError:pass
  else:raise AssertionError('wrong trajectory accepted')
with tempfile.TemporaryDirectory() as d:
 results,prior=build(False);results['1']['I1']=results['1']['I1']+1
 try:save(Path(d)/'bad',identity,results,targets,prefix,readout,4,3,prior,guard)
 except ValueError:pass
 else:raise AssertionError('identity mismatch')
print('LOCAL_ARTIFACT_REPLAY_CPU_PASS; no CUDA acceptance')
