python3 - <<'PY'
from pathlib import Path
import hashlib,json,os,struct
root=Path(os.environ.get('SHARD_ROOT','/home/luansiyu/loop-latent/orchestration_runs/run-20261107T053919Z-6d01138e/shards'))
stems={'train':'e7fab80fdd9f1772c5a36c12d7cdeab1e00531e777f81957f10b84b2f01669ef','validation':'8958656ab3c135446ce8fa1443c3a674f04781eb46c2edb7241f9797260a8e48','test':'494dd9ff934b79a132c71d5f91cb801e90865a98829c57618b94c59208186684'}
for split,stem in stems.items():
 p=root/split/(stem+'.safetensors'); j=p.with_suffix('.json')
 with p.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));base=8+n
  for suffix in ('loop1_layer00','loop4_layer23'):
   hashes=[]
   for c in range(4):
    key=f'candidate{c}_{suffix}';a,b=h[key]['data_offsets'];f.seek(base+a);hashes.append(hashlib.sha256(f.read(b-a)).hexdigest())
   print(split,suffix,'equal_all',len(set(hashes))==1,'distinct',len(set(hashes)),'sample_sha256',hashes[0])
 d=json.loads(j.read_text()); rec=d['records'];print(split,'records_type',type(rec).__name__,'count',len(rec))
 print(split,'shard',p.name,'recorded_tensor_sha256',d.get('tensor_sha256'),'metadata_sha256',hashlib.sha256(j.read_bytes()).hexdigest())
 if isinstance(rec,list) and rec: print('record_keys',list(rec[0])[:30])
 if isinstance(rec,list) and rec:
  for field in ('input_ids_sha256','captured_positions','effective_position_ids'):
   values=[json.dumps(r.get(field),sort_keys=True) for r in rec];print(split,field,'equal_all',len(set(values))==1)
PY
