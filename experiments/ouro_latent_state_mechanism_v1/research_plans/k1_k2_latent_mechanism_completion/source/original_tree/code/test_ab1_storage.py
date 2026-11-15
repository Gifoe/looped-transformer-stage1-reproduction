import tempfile,pathlib,json,gzip,hashlib,torch
from safetensors.torch import load
from ab1_storage import serialize_exclusive
with tempfile.TemporaryDirectory() as t:
 x={'E':torch.arange(48,dtype=torch.bfloat16).reshape(1,8,6),'Y':torch.tensor([1.25],dtype=torch.float64)};r=serialize_exclusive(t,'x',{'status':'synthetic'},x);d=pathlib.Path(t);m=json.loads(next(d.glob('*.json')).read_text());raw=gzip.decompress((d/m['tensor_file']).read_bytes());z=load(raw);assert all(torch.equal(z[k],v) for k,v in x.items()) and hashlib.sha256(raw).hexdigest()==r['tensor_raw_sha256']
 try:serialize_exclusive(t,'x',{},x)
 except FileExistsError:pass
 else:raise AssertionError('overwrite accepted')
print('PASS synthetic lossless gzip byte hashes and immutable duplicate refusal; no model data')
