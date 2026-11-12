"""Independent disk-to-CUDA score replay for exactly 3 probe cases."""
import json,sys,hashlib
from pathlib import Path
import torch
from safetensors.torch import load_file
from technical import score
from source_input import sha
if not __debug__:raise RuntimeError('optimized Python forbidden')
def main():
 out=Path(sys.argv[1]);receipt=json.loads((out/'probe_pending.json').read_text());rows=[]
 if receipt['status']!='PENDING_REPLAY' or len(receipt['questions'])!=3:raise ValueError('probe receipt')
 for row in receipt['questions']:
  mp=out/(row['label']+'.json');tp=out/(row['label']+'.safetensors')
  if sha(mp)!=row['metadata_sha256'] or sha(tp)!=row['tensor_sha256']:raise ValueError('artifact hash')
  meta=json.loads(mp.read_text());t=load_file(str(tp))
  if set(t)!=set(meta['scores']):raise ValueError('keys')
  for k,z in t.items():
   if z.dtype!=torch.bfloat16 or z.shape!=(1,49152) or not torch.isfinite(z).all() or score(z.to('cuda'),meta['targets'])!=meta['scores'][k]:raise ValueError('CUDA replay '+k)
  rows.append({'label':row['label'],'id':meta['id'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'conditions':sorted(t)})
 result={'status':'PROBE_CUDA_REPLAY_PASSED','questions':rows,'pending_sha256':sha(out/'probe_pending.json'),'script_sha256':sha(Path(__file__)),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()}
 with (out/'replay.json').open('x') as f:json.dump(result,f,sort_keys=True,allow_nan=False);f.write('\n')
 print('PROBE_CUDA_REPLAY_PASSED',flush=True)
if __name__=='__main__':main()
