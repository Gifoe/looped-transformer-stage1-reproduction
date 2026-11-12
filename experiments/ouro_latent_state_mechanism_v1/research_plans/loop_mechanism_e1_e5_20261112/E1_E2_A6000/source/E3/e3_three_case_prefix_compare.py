"""Three technical cases only: inspect actual prefix K/V identity across input lengths."""
import json,gzip,hashlib,datetime,os
from pathlib import Path
import torch
from safetensors.torch import load_file
torch.set_num_threads(1)
R=Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';P=B/'E1/technical_probe';REF=R/'research_plans/natural_prefix_identity_20261111/reference_prefix_ids.json';CAT=B/'results/compact_input_catalog.jsonl.gz'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
if not __debug__:raise RuntimeError('optimized Python forbidden')
assert sha(REF)=='e984a118a098923ba87fab4ddd4a26d1f9da8c5dc9767c5ab073352cd3dfbf67' and sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d';ref=json.loads(REF.read_text())['input_ids'];pending=json.loads((P/'pending.json').read_text());rep=json.loads((P/'replay.json').read_text());assert rep['pending_sha256']==sha(P/'pending.json') and rep['status']=='E1_THREE_CASE_CUDA_SCORE_AND_ENDPOINT_REPLAY_PASSED'
with gzip.open(CAT,'rt') as f:cat={(r['split'],r['id']):r for l in f if (r:=json.loads(l))['split']=='test'}
cases=[]
for i in range(3):
 mp=P/(str(i)+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==pending['questions'][i]['metadata_sha256'] and sha(tp)==pending['questions'][i]['tensor_sha256'];m=json.loads(mp.read_text());a=cat[('test',m['id'])];assert a['input_ids'][:1839]==ref and a['input_ids_sha256']==m['input_ids_sha256'];t=load_file(str(tp));cases.append((m,t,len(a['input_ids'])))
def thash(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def compare(x,y):
 assert x.shape==y.shape and x.dtype==y.dtype==torch.bfloat16;d=x.double()-y.double();norm=float(torch.linalg.vector_norm(x.double()));return {'bitwise_equal':torch.equal(x,y),'left_sha256':thash(x),'right_sha256':thash(y),'max_absolute':float(d.abs().max()),'difference_l2':float(torch.linalg.vector_norm(d)),'relative_l2_to_left':float(torch.linalg.vector_norm(d))/norm if norm else None}
out={'status':'THREE_CASE_PREFIX_KV_NUMERICAL_COMPARISON','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Three test technical inputs only; same prefix IDs, actual historical SDPA K/V; not a 64-cohort invariance result or a claim about future-information flow. A/B same length, C different length.','source_pending_sha256':sha(P/'pending.json'),'source_replay_sha256':sha(P/'replay.json'),'prefix_reference_sha256':sha(REF),'catalog_sha256':sha(CAT),'pairs':{}}
for ia,ib,name in ((0,1,'A_B_same_length'),(0,2,'A_C_different_length')):
 a,ta,la=cases[ia];b,tb,lb=cases[ib];assert (la==lb)==(name=='A_B_same_length');val={'left':a['id'],'right':b['id'],'left_length':la,'right_length':lb,'comparisons':{}}
 for u in (0,1):
  for n in ('k','v'):
   x=ta[f'natural_u{u}_{n}'][...,:1839,:];y=tb[f'natural_u{u}_{n}'][...,:1839,:];val['comparisons'][f'loop{u+1}_{n}']=compare(x,y)
  # subtract original BF16 values in FP64; compare raw deltas without re-rounding
 da=ta['natural_u1_v'][...,:1839,:].double()-ta['natural_u0_v'][...,:1839,:].double();db=tb['natural_u1_v'][...,:1839,:].double()-tb['natural_u0_v'][...,:1839,:].double();dv=da-db;dn=float(torch.linalg.vector_norm(da));val['comparisons']['delta_v']={'bitwise_equal':torch.equal(da,db),'left_sha256':hashlib.sha256(da.contiguous().numpy().tobytes()).hexdigest(),'right_sha256':hashlib.sha256(db.contiguous().numpy().tobytes()).hexdigest(),'max_absolute':float(dv.abs().max()),'difference_l2':float(torch.linalg.vector_norm(dv)),'relative_l2_to_left':float(torch.linalg.vector_norm(dv))/dn if dn else None};out['pairs'][name]=val
out['script_sha256']=sha(Path(__file__));path=B/'E3/three_case_prefix_kv_comparison.json'
with path.open('x') as f:json.dump(out,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
print('THREE_CASE_PREFIX_KV_COMPLETE',sha(path),flush=True)
