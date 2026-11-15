"""AB1 lossless immutable question storage; gzip raw tensor + archive SHA."""
import gzip,hashlib,json,os,uuid
from pathlib import Path
import torch
from safetensors.torch import save,load
if not __debug__:raise RuntimeError('optimized Python forbidden')
def digest(b):return hashlib.sha256(b).hexdigest()
def serialize_exclusive(directory,ident,metadata,tensors):
 d=Path(directory);d.mkdir(parents=True,exist_ok=True);stem=hashlib.sha256(ident.encode()).hexdigest();tp=d/(stem+'.safetensors.gz');mp=d/(stem+'.json')
 if tp.exists() or mp.exists():raise FileExistsError('existing pair or orphan; independently reconcile, do not overwrite')
 values={k:v.detach().cpu().contiguous().clone() for k,v in tensors.items()};raw=save(values);back=load(raw);assert set(back)==set(values) and all(torch.equal(back[k],v) for k,v in values.items());archive=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(archive)==raw
 meta=dict(metadata);meta.update(id=ident,raw_tensor_sha256=digest(raw),archive_sha256=digest(archive),archive_bytes=len(archive),raw_bytes=len(raw),tensor_file=tp.name);mb=(json.dumps(meta,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();salt=uuid.uuid4().hex;ta=d/(stem+'.'+salt+'.tensor.partial');ma=d/(stem+'.'+salt+'.meta.partial')
 # Persist and validate before making either final link. A crash between links is
 # an explicit orphan, not accepted as completed or silently overwritten.
 with ta.open('xb') as f:f.write(archive);f.flush();os.fsync(f.fileno())
 with ma.open('xb') as f:f.write(mb);f.flush();os.fsync(f.fileno())
 assert digest(ta.read_bytes())==meta['archive_sha256'] and digest(gzip.decompress(ta.read_bytes()))==meta['raw_tensor_sha256'] and json.loads(ma.read_text())==meta
 os.link(ta,tp);os.link(ma,mp);ta.unlink();ma.unlink()
 return {'id':ident,'metadata_sha256':digest(mb),'tensor_archive_sha256':meta['archive_sha256'],'tensor_raw_sha256':meta['raw_tensor_sha256'],'tensor_archive_bytes':len(archive)}
