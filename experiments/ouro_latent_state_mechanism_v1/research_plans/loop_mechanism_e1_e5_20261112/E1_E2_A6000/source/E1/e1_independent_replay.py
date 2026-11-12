"""Independent CUDA scoring and old-corner ID/restore review of three E1 technical cases."""
import json,hashlib,torch,os,fcntl
from pathlib import Path
from safetensors.torch import load_file
from e1_control_core import controls
from e1_donor_pool import Pool,OLD,sha
if not __debug__:raise RuntimeError('optimized Python forbidden')
B=Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E1/technical_probe');P=B/'pending.json';fd=os.open('/home/luansiyu/loop-latent/ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);pending_sha=sha(P);p=json.loads(P.read_text());assert all(sha(Path(path))==h for path,h in p['source_hashes'].items());assert p['status']=='E1_THREE_CASE_PENDING_INDEPENDENT_REPLAY' and p['count']==3 and torch.cuda.is_available() and torch.__version__==p['runtime']['torch'] and torch.version.cuda==p['runtime']['cuda'] and torch.cuda.get_device_name()==p['runtime']['gpu'];pool=Pool();questions=[]
for i in range(3):
 mp=B/(str(i)+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==p['questions'][i]['metadata_sha256'] and sha(tp)==p['questions'][i]['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));ident=m['id'];key=hashlib.sha256(ident.encode()).hexdigest();op=OLD/'test'/(key+'.safetensors');assert sha(op)==m['old_eight_tensor_sha256'];old=load_file(str(op));count=0
 for q in (1,2):
  cm=m['conditions'][str(q)];ss=[old[f'G{q}{j}{k}_sdpa_R'] for j,k in ((1,1),(1,2),(2,1),(2,2))];candidate=controls(*ss,torch.ones_like(ss[3],dtype=torch.float64));donor=None
  if candidate['delta_norm']!=0:donor,d=pool.select('test',ident,m['input_ids_sha256'],q);assert d==cm['donor']
  C=controls(*ss,donor)
  for name in ('ADD_exact_fp64','cast_residual_fp64','delta_fp64','restore'):assert torch.equal(C[name],t[f'q{q}_{name}'])
  for name in ('ID','ADD','ADD_NORM','UNRELATED','restore'):
   if C[name] is None:assert name=='ADD_NORM' and cm['ADD_NORM_status']=='UNDEFINED';continue
   z=t[f'q{q}_{name}_logits'];a=t[f'q{q}_{name}_attention_localnorm'];assert z.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and torch.isfinite(z).all() and torch.equal(C[name],t[f'q{q}_{name}_injected_sdpa_R']);lp=z.cuda().float().log_softmax(-1);score=[float(lp.gather(-1,torch.tensor([[v]],device='cuda')).sum()) for v in m['targets']];assert score==cm[name]['scores']
   if name in ('ID','restore'):assert torch.equal(z,old[f'G{q}22_logits']) and torch.equal(a,old[f'G{q}22_a_R'])
   count+=1
 questions.append({'id':ident,'conditions_scored':count,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)})
assert sha(P)==pending_sha and all(sha(Path(path))==h for path,h in p['source_hashes'].items())
out={'status':'E1_THREE_CASE_CUDA_SCORE_AND_ENDPOINT_REPLAY_PASSED','pending_sha256':sha(P),'questions':questions,'count':3,'scope':'Separate CUDA scoring/replay process; shares frozen control mathematics, not an independently implemented model forward.'};q=B/'replay.json'
with q.open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('E1_REPLAY_PASSED',sha(q),flush=True);os.close(fd)
