import tempfile,torch,json
from pathlib import Path
from types import SimpleNamespace
from artifacts import save,replay
z=torch.tensor([[1.,2.,3.,4.]],dtype=torch.bfloat16);a=torch.ones((1,5,3),dtype=torch.bfloat16);results={}
for r in ('1','2'):
 results[r]={k:z.clone() for k in ('B1','B2','I1','I2','U1','U2')};results[r].update({'capture1':SimpleNamespace(branch=a),'capture2':SimpleNamespace(branch=a.clone())})
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'q';identity={'id':'test','source_sha':'f'*64};targets=[1,2];guard=lambda:None
 save(p,identity,results,targets,[4],4,3,guard);assert replay(p,identity,targets,[4],4,3,'cpu',guard)['status']=='ZERO_SCORE_DIFFERENCE'
 try:save(p,identity,results,targets,[4],4,3,guard)
 except FileExistsError:pass
 else:raise AssertionError('overwrite')
 f=p/'values.safetensors';v=bytearray(f.read_bytes());v[-1]^=1;f.write_bytes(v)
 try:replay(p,identity,targets,[4],4,3,'cpu',guard)
 except ValueError:pass
 else:raise AssertionError('corrupt accepted')
print('LOCAL_ARTIFACT_REPLAY_CPU_PASS; no CUDA acceptance')
