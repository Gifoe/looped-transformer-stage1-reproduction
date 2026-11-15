"""Immutable lossless tensor chunks. path in load(meta,path) is save directory.
Four full sequence tensors split at axis -2; every other tensor saved intact.
Single writer required; orphan chunks are retained and never signify completion.
"""
import gzip, hashlib, json, os
from pathlib import Path
import torch
from safetensors.torch import save as st_save, load as st_load
FULL_KEYS = {'source_E','source_B','native_Q9_full','U_Q9_full'}
def sha(b): return hashlib.sha256(b).hexdigest()
def rawbytes(t): return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
def spec(t): return {'shape':list(t.shape),'dtype':str(t.dtype),'byte_sha256':sha(rawbytes(t))}
def persist(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_bytes()!=b: raise RuntimeError('content-address collision or corrupt existing file: '+str(p))
        return
    with p.open('xb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
def chunk(d,t,shared):
    raw=st_save({'value':t.contiguous().clone()}); packed=gzip.compress(raw,compresslevel=6,mtime=0)
    rel=Path('shared' if shared else 'chunks')/(sha(raw)+'.safetensors.gz')
    persist(d/rel,packed)
    return dict(file=rel.as_posix(),raw_sha256=sha(raw),archive_sha256=sha(packed),raw_bytes=len(raw),archive_bytes=len(packed),**spec(t))
def readchunk(d,c):
    p=Path(c['file'])
    if p.is_absolute() or '..' in p.parts: raise ValueError('unsafe chunk path')
    b=(d/p).read_bytes()
    if sha(b)!=c['archive_sha256'] or len(b)!=c['archive_bytes']: raise ValueError('chunk archive integrity')
    raw=gzip.decompress(b)
    if sha(raw)!=c['raw_sha256'] or len(raw)!=c['raw_bytes']: raise ValueError('chunk raw integrity')
    v=st_load(raw)
    if set(v)!={'value'} or spec(v['value'])!={k:c[k] for k in ('shape','dtype','byte_sha256')}: raise ValueError('chunk tensor integrity')
    return v['value']
def load(meta,path):
    d=Path(path)
    if isinstance(meta,(str,Path)): meta=json.loads(Path(meta).read_text())
    if meta['codec']!='ab1_prefix_cas_v1': raise ValueError('unknown codec')
    result={}
    for k,v in meta['tensor_manifest'].items():
        parts=[readchunk(d,c) for c in v['chunks']]
        t=torch.cat(parts,dim=v['axis']) if v['axis'] is not None else parts[0]
        if spec(t)!={n:v[n] for n in ('shape','dtype','byte_sha256')}: raise ValueError('reconstructed tensor integrity: '+k)
        result[k]=t
    canonical={k:result[k] for k in sorted(result)}
    if sha(st_save(canonical))!=meta['raw_tensor_sha256']: raise ValueError('canonical full safetensors integrity')
    return result
def save(directory,id,metadata,tensors,prefix_len):
    d=Path(directory); d.mkdir(parents=True,exist_ok=True)
    stem=sha(id.encode()); mp=d/(stem+'.json')
    if mp.exists(): raise FileExistsError('existing committed case')
    if not isinstance(prefix_len,int) or prefix_len<0: raise ValueError('prefix length')
    values={k:t.detach().cpu().contiguous().clone() for k,t in sorted(tensors.items())}
    manifests={}
    for k,t in values.items():
        axis=None
        if k in FULL_KEYS:
            if t.ndim<2 or prefix_len>t.shape[-2]: raise ValueError('full sequence shape/prefix: '+k)
            axis=t.ndim-2
            parts=[chunk(d,t.narrow(axis,0,prefix_len),True),chunk(d,t.narrow(axis,prefix_len,t.shape[axis]-prefix_len),False)]
        else: parts=[chunk(d,t,False)]
        manifests[k]=dict(**spec(t),axis=axis,chunks=parts)
    raw=st_save(values)
    meta=dict(metadata); meta.update(id=id,codec='ab1_prefix_cas_v1',prefix_len=prefix_len,tensor_manifest=manifests,raw_tensor_sha256=sha(raw),raw_bytes=len(raw))
    back=load(meta,d)
    if set(back)!=set(values) or any(rawbytes(back[k])!=rawbytes(t) for k,t in values.items()): raise ValueError('writer bitwise verification failed')
    mb=(json.dumps(meta,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    # Exclusive final creation: partial writes fail JSON/hash verification and require reconciliation.
    with mp.open('xb') as f: f.write(mb); f.flush(); os.fsync(f.fileno())
    if mp.read_bytes()!=mb: raise ValueError('manifest write verification')
    return {'id':id,'metadata_file':mp.name,'metadata_sha256':sha(mb),'tensor_raw_sha256':sha(raw),'tensor_raw_bytes':len(raw),'codec':meta['codec']}
