"""Extract paired loop R-site latent transitions from audited natural branch tensors."""
import os,json,hashlib,gzip,datetime
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
from safetensors.torch import save_file
ROOT=Path('/home/luansiyu/loop-latent'); B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';O=B/'E2';BR=ROOT/'research_plans/natural_branch_full_run_20261111'; LC=ROOT/'research_plans/NATURAL_L18_INDEPENDENT_COVERAGE_20261111.json'; GEO=ROOT/'research_plans/natural_r_geometry_20261111'; MODEL=ROOT/'model/model.safetensors';CAT=B/'results/compact_input_catalog.jsonl.gz'
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def norm(x):return float(np.linalg.norm(x))
def pair(u,v):
 d=v-u; nu,nv=norm(u),norm(v);return {'k1_l2':nu,'k2_l2':nv,'delta_l2':norm(d),'cos':float(u@v/(nu*nv)) if nu and nv else None,'relative_delta':float(norm(d)/nu) if nu else None}
assert sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d' and sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
l=json.loads(LC.read_text()); assert l['count']==2590 and l['status']=='INDEPENDENT_L18_CUDA_AND_SOURCE_COVERAGE_PASSED';links={(x['split'],x['id']):x for x in l['questions']};assert len(links)==2590
with gzip.open(CAT,'rt') as f:catalog=[json.loads(x) for x in f]
assert len(catalog)==2590 and {(r['split'],r['id']) for r in catalog}==set(links)
with safe_open(str(MODEL),framework='pt',device='cpu') as f:
 gamma=f.get_tensor('model.norm.weight').float().numpy().astype('float64');W=f.get_tensor('lm_head.weight').float().numpy().astype('float64')
config=json.loads((ROOT/'model/config.json').read_text());eps=config['rms_norm_eps'];rows=[];all_states={};n=0
for split in ('train','validation','test'):
 geo={}
 with (GEO/(split+'_rows.jsonl')).open() as f:
  for line in f:
   g=json.loads(line);geo[g['id']]=g
 for r in [r for r in catalog if r['split']==split]:
  ident=r['id'];q=links[(split,ident)];key=split+'_'+hashlib.sha256(ident.encode()).hexdigest();mp=BR/(key+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==q['branch_metadata_sha256'] and sha(tp)==q['branch_tensor_sha256'] and q['natural_metadata_sha256']==r['metadata_sha256'] and q['natural_tensor_sha256']==r['historical_natural_tensor_sha256'];m=json.loads(mp.read_text());g=geo[ident];assert m['id']==ident and m['split']==split and m['R_absolute']==r['R'] and m['targets']==r['targets'] and m['gold']==r['gold_index'] and m['prior_scores']==r['native_scores'][:2] and g['metadata_sha256']==r['metadata_sha256'] and g['tensor_sha256']==r['historical_natural_tensor_sha256'];targets=r['targets'];gold=r['gold_index'];wrong=[i for i in range(len(targets)) if i!=gold];states={};features=[]
  with safe_open(str(tp),framework='pt',device='cpu') as f:
   for u in (0,1):
    prev=None
    for i in range(24):
     v={k:f.get_tensor(f'u{u}_layer{i:02d}_{k}') for k in ('x','a','y','b','z')};assert all(x.dtype==torch.bfloat16 and tuple(x.shape)==(1,2048) and torch.isfinite(x).all() for x in v.values());assert torch.equal((v['x']+v['a']).to(torch.bfloat16),v['y']) and torch.equal((v['y']+v['b']).to(torch.bfloat16),v['z']);assert prev is None or torch.equal(v['x'],prev);prev=v['z'];states[(u,i)]={k:x.float().numpy().astype('float64').ravel() for k,x in v.items()}
     if i==17:all_states[f'{key}_u{u}_l18_input']=v['x'].clone()
    post=f.get_tensor(f'loop{u+1}_post_norm');assert post.dtype==torch.bfloat16 and tuple(post.shape)==(1,2048);all_states[f'{key}_u{u}_post_norm']=post.clone()
    if u==0:post1=post.clone()
   assert torch.equal(f.get_tensor('u1_layer00_x'),post1)
  for i in range(24):
   for site in ('x','a','y','b','z'):
    u,v=states[(0,i)][site],states[(1,i)][site];item=pair(u,v)
    if site in ('x','y','z'):
     def lens(h):
      z=W[targets]@(gamma*h/np.sqrt(np.mean(h*h)+eps));return [float(z[gold]-z[j]) for j in wrong]
     g1,g2=lens(u),lens(v);item['diagnostic_candidate_gap_k1']=g1;item['diagnostic_candidate_gap_k2']=g2;item['diagnostic_gap_delta_mean']=float(np.mean(np.array(g2)-np.array(g1)))
    features.append({'layer_index':i,'site':site,**item})
  rows.append({'id':ident,'split':split,'demo':g['demo'],'group':g['group'],'natural_no_top_tie':g['natural_no_top_tie'],'gold':gold,'targets':targets,'native_scores':r['native_scores'],'branch_metadata_sha256':q['branch_metadata_sha256'],'branch_tensor_sha256':q['branch_tensor_sha256'],'natural_metadata_sha256':r['metadata_sha256'],'historical_natural_tensor_sha256':r['historical_natural_tensor_sha256'],'features':features});n+=1
  if n%100==0:print('E2_STATES',n,flush=True)
assert n==2590 and sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
out=O/'E2_latent_transition_rows.jsonl.gz'
with gzip.open(out,'xt') as f:
 for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
tensors=O/'E2_fixed_probe_states.safetensors';save_file(all_states,str(tensors))
receipt={'status':'E2_NATURAL_R_STATE_EXTRACTION_COMPLETE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':n,'catalog_sha256':sha(CAT),'l18_coverage_sha256':sha(LC),'model_sha256':sha(MODEL),'script_sha256':sha(Path(__file__)),'rows_sha256':sha(out),'probe_states_sha256':sha(tensors),'scope':'Only saved R token; original deleted natural whole tensors are historical links, not reread. W/RMS candidate gaps are offline diagnostic lens, not original production predictions. Gold used only for descriptive metrics.'}
p=O/'E2_state_extract.receipt.json'
with p.open('x') as f:json.dump(receipt,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('E2_COMPLETE',json.dumps(receipt),flush=True)
