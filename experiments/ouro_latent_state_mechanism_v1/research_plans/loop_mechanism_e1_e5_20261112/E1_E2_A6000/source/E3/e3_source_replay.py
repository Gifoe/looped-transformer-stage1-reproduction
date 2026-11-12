"""Independent CUDA source-control, saved scores and old endpoint replay for E3 technical probe."""
import json,hashlib,fcntl,os,torch
from pathlib import Path
from safetensors.torch import load_file
from e3_source_control_v2 import build
ROOT=Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';OUT=B/'E3/source_technical';PRIOR=B/'E1/technical_probe';SEL=['ACTAAP_2013_7_16','Mercury_405804','MSA_2012_8_30'];OLD=ROOT/'research_plans/natural_qkv_eight_corner_20261112';P=OUT/'pending.json'
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
fd=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);psha=sha(P);p=json.loads(P.read_text());assert p['status']=='E3_SOURCE_THREE_CASE_PENDING_REPLAY' and p['count']==3 and all(sha(Path(k))==v for k,v in p['source_hashes'].items()) and torch.cuda.is_available() and torch.__version__==p['runtime']['torch'] and torch.version.cuda==p['runtime']['cuda'] and torch.cuda.get_device_name()==p['runtime']['gpu'];pp=json.loads((PRIOR/'pending.json').read_text());out=[]
for i in range(3):
 mp=OUT/(str(i)+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==p['questions'][i]['metadata_sha256'] and sha(tp)==p['questions'][i]['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));pm=PRIOR/(str(i)+'.json');pt=pm.with_suffix('.safetensors');assert sha(pm)==pp['questions'][i]['metadata_sha256'] and sha(pt)==pp['questions'][i]['tensor_sha256'];prior=load_file(str(pt));pmj=json.loads(pm.read_text());assert m['id']==pmj['id']==SEL[i] and m['split']==pmj['split']=='test' and m['R']==pmj['R'] and m['targets']==pmj['targets'] and m['input_ids_sha256']==pmj['input_ids_sha256'];neednames={'ID','K_Q','V_Q','KV_Q','KV_P','K_Q_NORM','V_Q_NORM','V_Q_ROLL'};assert set(m['conditions'])=={'1','2'} and all(set(m['conditions'][str(q)])==neednames for q in (1,2));expected={f'q{q}_{name}_{suffix}' for q in (1,2) for name in neednames if m['conditions'][str(q)][name]['control']['status']=='OK' for suffix in ('logits','sdpa_R','a_R')};assert set(t)==expected;key=hashlib.sha256(m['id'].encode()).hexdigest();op=OLD/'test'/(key+'.safetensors');om=op.with_suffix('.json');assert sha(op)==m['old_tensor_sha256'] and sha(om)==m['old_metadata_sha256'];oldm=json.loads(om.read_text());assert oldm['id']==m['id'] and oldm['R']==m['R'] and oldm['targets']==m['targets'];old=load_file(str(op));raw={f'{n}{u+1}':prior[f'natural_u{u}_{n}'].cuda() for n in ('k','v') for u in (0,1)}
 for depth in (1,2):
  assert m['captures'][str(depth)]==oldm['capture_receipts'][str(depth)]['qkv']
  for n in ('k','v'):assert thash(raw[n+str(depth)])==m['captures'][str(depth)]['qkv'][n]['sha256']
 count=0
 for q in (1,2):
  for name in ('ID','K_Q','V_Q','KV_Q','KV_P','K_Q_NORM','V_Q_NORM','V_Q_ROLL'):
   k,v,c=build(raw['k1'],raw['k2'],raw['v1'],raw['v2'],m['R'],1839,name);cm=m['conditions'][str(q)][name];assert cm['control']==c
   if k is None:continue
   assert thash(k)==cm['k_sha256'] and thash(v)==cm['v_sha256'];tag=f'q{q}_{name}';z=t[tag+'_logits'];s=t[tag+'_sdpa_R'];a=t[tag+'_a_R'];assert z.dtype==s.dtype==a.dtype==torch.bfloat16 and tuple(z.shape)==(1,49152) and tuple(s.shape)==(1,16,128) and tuple(a.shape)==(1,2048) and torch.isfinite(z).all() and torch.isfinite(s).all() and torch.isfinite(a).all() and thash(z)==cm['logits_sha256'] and thash(s)==cm['sdpa_R_sha256'] and thash(a)==cm['a_R_sha256'] and cm['outside_R_unchanged'];lp=z.cuda().float().log_softmax(-1);scores=[float(lp.gather(-1,torch.tensor([[j]],device='cuda')).sum()) for j in m['targets']];assert scores==cm['scores']
   if name=='ID':assert scores==oldm['scores'][f'G{q}22'] and torch.equal(z,old[f'G{q}22_logits']) and torch.equal(s,old[f'G{q}22_sdpa_R']) and torch.equal(a,old[f'G{q}22_a_R'])
   count+=1
 out.append({'id':m['id'],'conditions_scored':count,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)})
assert sha(ROOT/'model/model.safetensors')=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af' and sha(P)==psha and all(sha(Path(k))==v for k,v in p['source_hashes'].items());r={'status':'E3_SOURCE_THREE_CASE_CUDA_REPLAY_PASSED','pending_sha256':psha,'count':3,'questions':out,'scope':'Independent saved-score CUDA replay and region K/V control recomputation from prior technical raw K/V; does not independently rerun full model forward or claim fake Q rows.'};pout=OUT/'replay.json'
with pout.open('x') as f:json.dump(r,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('E3_SOURCE_REPLAY_PASSED',sha(pout),flush=True);os.close(fd)
