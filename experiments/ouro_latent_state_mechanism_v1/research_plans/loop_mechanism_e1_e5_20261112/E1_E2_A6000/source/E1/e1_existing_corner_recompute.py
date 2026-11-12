"""Independent CPU E1 reanalysis of audited saved eight-corner s_R and a_R."""
import os,json,hashlib,gzip,datetime
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
ROOT=Path('/home/luansiyu/loop-latent');OLD=ROOT/'research_plans/natural_qkv_eight_corner_20261112';OUT=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E1'; MODEL=ROOT/'model/model.safetensors'
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def rms(v):return float(np.linalg.norm(v))
def cos(a,b):
 d=rms(a)*rms(b);return float(a@b/d) if d else None
assert sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
with safe_open(str(MODEL),framework='pt',device='cpu') as f:
 O=f.get_tensor('model.layers.17.self_attn.o_proj.weight').float().numpy().astype('float64');gamma=f.get_tensor('model.norm.weight').float().numpy().astype('float64'); W=f.get_tensor('lm_head.weight').float().numpy().astype('float64')
weights={'oproj_tensor_sha256':hashlib.sha256(O.astype('float32').tobytes()).hexdigest(),'final_norm_tensor_sha256':hashlib.sha256(gamma.astype('float32').tobytes()).hexdigest(),'head_tensor_sha256':hashlib.sha256(W.astype('float32').tobytes()).hexdigest()}
results=[];source={};worst=0.
for split,n in [('train',1119),('validation',299),('test',1172)]:
 cov=OLD/(split+'_independent_cuda_coverage.json');summary=OLD/(split+'_analysis/summary.json');rows=OLD/(split+'_analysis/rows.jsonl.gz');c=json.loads(cov.read_text());sm=json.loads(summary.read_text());assert c['count']==sm['count']==n and sm['coverage_sha256']==sha(cov) and sm['rows_sha256']==sha(rows);source[split]={'coverage_sha256':sha(cov),'summary_sha256':sha(summary),'rows_sha256':sha(rows)};ledger={x['id']:x for x in c['questions']}
 with gzip.open(rows,'rt') as f: rr=[json.loads(x) for x in f]
 assert len(rr)==len(ledger)==n and {r['id'] for r in rr}==set(ledger)
 for r in rr:
  key=hashlib.sha256(r['id'].encode()).hexdigest();tp=OLD/split/(key+'.safetensors');assert sha(tp)==ledger[r['id']]['tensor_sha256']==r['tensor_sha256'];v={}
  with safe_open(str(tp),framework='pt',device='cpu') as f:
   for u in ('sdpa_R','a_R'):
    x={f'G{i}{j}{k}':f.get_tensor(f'G{i}{j}{k}_{u}').float().numpy().astype('float64').ravel() for i in (1,2) for j in (1,2) for k in (1,2)};v[u]={}
    for i in (1,2):
     d=x[f'G{i}22']-x[f'G{i}21']-x[f'G{i}12']+x[f'G{i}11'];expected=r['vector_differences'][u]['differences'][f'I_Q{i}']['l2'];worst=max(worst,abs(rms(d)-expected));assert np.isclose(rms(d),expected,rtol=1e-12,atol=1e-12);v[u][i]=d
   targets=r['targets'];gold=r['gold'];wrong=[j for j in range(len(targets)) if j!=gold];wg=(W[targets[gold]]-np.mean(W[[targets[j] for j in wrong]],axis=0))*gamma;fixed=(W[targets[gold]]-W[targets[r['fixed_wrong']]])*gamma;items={}
   for i in (1,2):
    s=v['sdpa_R'][i];a=v['a_R'][i];linear=O@s;items[str(i)]={'sdpa_interaction_l2':rms(s),'oproj_linear_interaction_l2':rms(linear),'actual_a_interaction_l2':rms(a),'linear_vs_actual_cos':cos(linear,a),'linear_wgamma_allwrong':float(wg@linear),'actual_a_wgamma_allwrong':float(wg@a),'linear_wgamma_fixedwrong':float(fixed@linear),'actual_a_wgamma_fixedwrong':float(fixed@a),'score_gap_interaction':r['gap_differences'][f'I_Q{i}']}
  results.append({'split':split,'id':r['id'],'demo':r['demo'],'group':r['group'],'natural_no_top_tie':r['natural_no_top_tie'],'source_tensor_sha256':r['tensor_sha256'],'Q':items})
  if len(results)%100==0:print('E1_PROJECTION',len(results),flush=True)
assert len(results)==2590 and sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
out=OUT/'E1_existing_corner_projection.jsonl.gz'
with gzip.open(out,'xt') as f:
 for r in results:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
receipt={'status':'AUDITED_OLD_CORNER_CPU_PROJECTION_COMPLETE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(results),'source':source,'weights':weights,'model_checkpoint_historical_sha256':'58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af','worst_saved_l2_disagreement':worst,'output_sha256':sha(out),'script_sha256':sha(Path(__file__)),'scope':'FP64 linear O projection and Wgamma diagnostic; actual saved a_R is post-localnorm, not reconstructed GPU o_proj or native final score; no QKV or softmax attention weights available.'}
p=OUT/'E1_existing_corner_projection.receipt.json'
with p.open('x') as f:json.dump(receipt,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('E1_COMPLETE',json.dumps(receipt),flush=True)
