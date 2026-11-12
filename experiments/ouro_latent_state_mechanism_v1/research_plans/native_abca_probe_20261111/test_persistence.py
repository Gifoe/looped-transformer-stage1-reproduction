import tempfile,torch
from pathlib import Path
from persistence import save_capture,compare_all
p=torch.zeros((1,3,4),dtype=torch.bfloat16);rope=torch.ones((1,5,4),dtype=torch.bfloat16);o={'states':[{'prefix':p,'source_device':'cpu'},{'prefix':p,'source_device':'cpu'}],'rope':[{'cos_sin':(rope,rope),'position_ids':torch.arange(5).unsqueeze(0),'source_devices':['cpu','cpu']}],'attention':[],'sdpa':[],'layers':[(0,0)],'kernel':'UNKNOWN'}
with tempfile.TemporaryDirectory() as d:
 for n in ('A1','B','C','A2'):save_capture(d,n,{'id':'a' if n in ('A1','A2') else n},o,[1.],torch.zeros(1,8,dtype=torch.bfloat16),'a'*64)
 v=compare_all(d);assert all(a['bitwise_equal'] for a in v.values())
 try:save_capture(d,'A1',{'id':'a'},o,[1.],torch.zeros(1,8,dtype=torch.bfloat16),'a'*64)
 except FileExistsError:pass
 else:raise AssertionError('overwrite')
 f=Path(d)/'B.safetensors';b=bytearray(f.read_bytes());b[-1]^=1;f.write_bytes(b)
 try:compare_all(d)
 except ValueError:pass
 else:raise AssertionError('corrupt')
print('PERSISTENCE_TOY_PASS; NOT GPU')
