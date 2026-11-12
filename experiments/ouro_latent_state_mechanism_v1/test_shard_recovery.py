"""Offline atomic shard recovery and corruption tests."""
import tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from cohort_runner import sha,write_json,complete_shard
base=Path(__file__).resolve().parents[3]
with tempfile.TemporaryDirectory(prefix='shard_check_',dir=base) as tmp:
 root=Path(tmp);meta=root/'a.json';payload=root/'a.safetensors';partial=root/'a.safetensors.partial';hashes={'source':'abc'}
 assert not complete_shard(meta,payload,'id',hashes)
 x=torch.tensor([[1.5,-2.25]],dtype=torch.bfloat16);save_file({'x':x},str(partial));assert torch.equal(load_file(str(partial))['x'],x)
 assert not complete_shard(meta,payload,'id',hashes) # interrupted before rename
 partial.replace(payload);assert not complete_shard(meta,payload,'id',hashes) # payload only must replay
 write_json(meta,{'id':'id','sha256':hashes,'tensor_sha256':sha(payload)});assert complete_shard(meta,payload,'id',hashes)
 for wrong in ({'source':'different'},):
  try:complete_shard(meta,payload,'id',wrong);raise AssertionError('stale accepted')
  except RuntimeError:pass
 with payload.open('ab') as f:f.write(b'corrupt')
 try:complete_shard(meta,payload,'id',hashes);raise AssertionError('corruption accepted')
 except RuntimeError:pass
print('PASS shard recovery, stale hash, corrupted shard and BF16 roundtrip')
