"""Independent CPU audit of all 64 value-position partition artifacts."""
import os,json,hashlib,gzip,datetime
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
from safetensors.torch import load_file
R=Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E1';O=B/'E3/discovery64';P=O/'receipt.json';SEL=B/'orders/E1_DISCOVERY64_FROZEN.json';A=E/'discovery64_cuda_audit.json';MODEL=R/'model/model.safetensors';OLD=R/'research_plans/natural_qkv_eight_corner_20261112'
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
receipt=json.loads(P.read_text());sel=json.loads(SEL.read_text());audit=json.loads(A.read_text());assert receipt['status']=='E3_DISCOVERY64_PARTITION_CAPTURE_PENDING_INDEPENDENT_AUDIT' and receipt['count']==64 and receipt['selection_sha256']==sha(SEL) and audit['status']=='DISCOVERY64_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED' and all(sha(Path(k))==v for k,v in receipt['source_hashes'].items()); ids={v['id']:g for g,vs in sel['groups'].items() for v in vs};assert len(ids)==64 and {x['id'] for x in receipt['questions']}==set(ids);oldledger={q['id']:q for q in audit['questions']};assert set(oldledger)==set(ids) and sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
with safe_open(str(MODEL),framework='pt',device='cpu') as f:Oproj=f.get_tensor('model.layers.17.self_attn.o_proj.weight').float().numpy().astype('float64');W=f.get_tensor('lm_head.weight').float().numpy().astype('float64');gamma=f.get_tensor('model.norm.weight').float().numpy().astype('float64')
rows=[];worst=0.
for x in receipt['questions']:
 ident=x['id'];key=hashlib.sha256(ident.encode()).hexdigest();mp=O/(key+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==x['metadata_sha256'] and sha(tp)==x['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));ep=E/'train'/(key+'.safetensors');em=ep.with_suffix('.json');assert sha(ep)==oldledger[ident]['tensor_sha256'] and sha(em)==oldledger[ident]['metadata_sha256'] and m['source_e1_tensor_sha256']==sha(ep) and m['source_e1_metadata_sha256']==sha(em);src=load_file(str(ep));oldmeta=OLD/'train'/(key+'.json');assert sha(oldmeta)==oldledger[ident]['old_metadata_sha256'];old=json.loads(oldmeta.read_text());gold=old['gold'];targets=old['targets'];wrong=[j for j in range(len(targets)) if j!=gold];fixed=old['fixed_wrong'];wg=(W[targets[gold]]-np.mean(W[[targets[j] for j in wrong]],axis=0))*gamma;wf=(W[targets[gold]]-W[targets[fixed]])*gamma;item={'id':ident,'group':ids[ident],'input_ids_sha256':m['input_ids_sha256'],'source_metadata_sha256':sha(mp),'source_tensor_sha256':sha(tp),'q':{}}
 assert set(t)=={f'q{q}_{k}' for q in (1,2) for k in ('P','Q','R','numeric_residual')}
 for qi in (1,2):
  parts={k:t[f'q{qi}_{k}'].numpy() for k in ('P','Q','R','numeric_residual')};assert all(x.dtype==np.float64 and x.shape==(1,16,128) and np.isfinite(x).all() for x in parts.values());d=src[f'q{qi}_delta_fp64'].numpy();err=float(np.max(np.abs(sum(parts.values())-d)));worst=max(worst,err);assert np.allclose(sum(parts.values()),d,atol=1e-10,rtol=1e-10)
  values={}
  for name,x in parts.items():
   v=Oproj@x.ravel();got={'vector_l2':float(np.linalg.norm(x)),'oproj_linear_l2':float(np.linalg.norm(v)),'wgamma_allwrong':float(wg@v),'wgamma_fixedwrong':float(wf@v)};ref=m['q'][str(qi)]['component_diagnostics'][name];assert all(np.isclose(got[k],ref[k],atol=1e-10,rtol=1e-10) for k in got);values[name]=got
  for name in ('A1','A2'):
   a=src[f'q{qi}_{name}_fp64'];assert a.dtype==torch.float64 and a.shape[0]==1 and a.shape[1]==16 and torch.isfinite(a).all() and (a>=0).all() and torch.allclose(a.sum(-1),torch.ones_like(a.sum(-1)),atol=1e-12,rtol=1e-12) and torch.count_nonzero(a[...,m['R']+1:])==0
  item['q'][str(qi)]=values
 rows.append(item)
assert len(rows)==64 and sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af' and all(sha(Path(k))==v for k,v in receipt['source_hashes'].items());rows_path=O/'audited_rows.jsonl.gz'
with gzip.open(rows_path,'xt') as f:
 for x in rows:f.write(json.dumps(x,sort_keys=True,allow_nan=False)+'\n')
out={'status':'E3_DISCOVERY64_INDEPENDENT_SAVED_PARTITION_AUDIT_PASSED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':64,'collector_receipt_sha256':sha(P),'selection_sha256':sha(SEL),'audited_rows_sha256':sha(rows_path),'script_sha256':sha(Path(__file__)),'max_partition_accounting_error':worst,'scope':'Independently checks saved partition vectors, candidate projections, previous A row legality and source hash links. Since full K/V were not archived, this audit cannot independently reconstruct A or per-token terms from raw K/V; natural recapture source hashes and collector evidence remain necessary.'};dest=O/'independent_audit.json'
with dest.open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('E3_DISCOVERY64_AUDIT_COMPLETE',sha(dest),flush=True)
