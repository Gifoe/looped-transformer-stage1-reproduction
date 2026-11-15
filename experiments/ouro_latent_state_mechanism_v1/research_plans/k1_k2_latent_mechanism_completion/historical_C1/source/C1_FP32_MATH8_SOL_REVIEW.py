"""Independent CPU read-only tensor/metadata review; never load a model or CUDA."""
import json,gzip,hashlib
from pathlib import Path
from fractions import Fraction
import torch
from safetensors.torch import load
R=Path('/root/autodl-tmp/ouro-k1-k2-mechanism');O=R/'numeric_fp32_deterministic_math_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
def thash(a):return hashlib.sha256(a.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tensors(m):
 p=R/m['tensor_file'];assert sha(p)==m['tensor_archive_sha256'];raw=gzip.decompress(p.read_bytes());assert hashlib.sha256(raw).hexdigest()==m['tensor_sha256'];return load(raw)
def sign(x):return int(x>0)-int(x<0)
def mean_gap(vals,g):
 v=[Fraction(float(x)) for x in vals];return v[g]-sum(x for i,x in enumerate(v) if i!=g)/(len(v)-1)
complete=read(O/'COMPLETE.json');assert complete['n']==8 and complete['status']=='COMPLETE'
freeze=read(O/'SOURCE_AND_SAMPLE_FREEZE.json');labels={q['id']:q.get('gold',q.get('gold_index')) for q in map(json.loads,Path('/root/autodl-tmp/ouro-e6a/next_step_train_diagnosis/TRAIN_ONLY_INPUTS.jsonl').read_text().splitlines())}
conditions=['K1','N','P','U','UX','UA','UXA'];contrasts={'N-U':{'N':1,'U':-1},'UX-U':{'UX':1,'U':-1},'N-UA':{'N':1,'UA':-1},'UA-U':{'UA':1,'U':-1},'N-UX':{'N':1,'UX':-1},'interaction':{'N':1,'UX':-1,'UA':-1,'U':1}}
rows=[]
for ident in freeze['ids']:
 stem=hashlib.sha256(ident.encode()).hexdigest();m=read(O/(stem+'.json'));t=tensors(m);refpath=R/'references/train'/(stem+'.json');b=read(refpath);bt=tensors(b);assert sha(refpath)==m['native_BF16_reference_json_sha256'] and b['input_ids_sha256']==m['input_ids_sha256'];g=labels[ident];assert g is not None
 assert all(v.device.type=='cpu' and torch.isfinite(v).all() for v in t.values());assert all(v.dtype==torch.float32 for k,v in t.items() if k.endswith('_logits'))
 for c,events in m['actual_SDPA_profiler_events'].items():assert any(e['operator']=='aten::_scaled_dot_product_attention_math' and e['count']==48 for e in events) and not any('efficient' in e['operator'] or 'flash' in e['operator'] for e in events)
 iden=m['source_identity'];assert iden['deterministic_algorithms'] and iden['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and not iden['TF32_matmul'] and not iden['TF32_cudnn']
 for i in range(9,16):
  d=t[f'd_FP32ref_{i:02d}'];assert bits(d,t[f'N_a_FP32_{i:02d}'].double()-t[f'P_a_FP32_{i:02d}'].double())
  for c in ['U','UX','UA','UXA']:
   p=f'{c}_site_{i:02d}_';before=t[p+'before'];exact=t[p+'exact_before_cast'];after=t[p+'after'];assert bits(exact,before.double()-d) and bits(after,exact.float()) and bits(t[p+'cast_residual'],after.double()-exact)
   if c!='U':
    for suffix in ['before','after','exact_before_cast','cast_residual','normalization_scale_fp64']:assert bits(t[p+suffix],t[f'U_site_{i:02d}_'+suffix])
 assert bits(t['UXA_logits'],t['N_logits'])
 for c in ['UX','UA','UXA']:
  for suffix in ['x_R','a_R','Q_R','K_R','V_R','O_R']:assert bits(t[c+'_'+suffix],t['U_'+suffix])
 for c,xb,ab in [('U','U','U'),('UX','N','U'),('UA','U','N'),('UXA','N','N')]:
  assert bits(t[c+'_y_R'],t[xb+'_x_R']+t[ab+'_a_R']) and bits(t[c+'_MLP_input_R'],t[c+'_y_R']) and bits(t[c+'_MLP_skip_R'],t[c+'_y_R']) and bits(t[c+'_after_mlp_R'],t[c+'_y_R']+t[c+'_b_R'])
 scores={};bf={};endpoints={}
 for c in conditions:
  z=t[c+'_logits'];sm=m['conditions'][c];assert thash(z)==sm['Rlogits_sha256'];vals=z[m['targets']].tolist();tie=[i for i,x in enumerate(vals) if x==max(vals)];assert vals==sm['candidate_logits'] and tie==sm['top_ties'] and tie[0]==sm['prediction'];lp=z.float().log_softmax(-1)[m['targets']].tolist();score_cpu_diff=max(abs(x-y) for x,y in zip(lp,sm['scores']));scores[c]=mean_gap(sm['scores'],g)
  if c in ['K1','N','P','U']:bm=b['conditions'][c];bz=bt[c+'_logits']
  else:
   bm=read(R/'treatments/train'/stem/(c+'.json'));bz=tensors(bm)['logits']
  assert thash(bz)==bm['Rlogits_sha256'];bv=bz[m['targets']].float().tolist();blp=bz.float().log_softmax(-1)[m['targets']].tolist();assert bv==bm['candidate_logits'];bf[c]=mean_gap(bm['scores'],g)
  endpoints[c]={'FP32_CPU_logsoftmax_max_abs_difference':score_cpu_diff,'BF16_CPU_logsoftmax_max_abs_difference':max(abs(x-y) for x,y in zip(blp,bm['scores'])),'FP32_prediction':sm['prediction'],'BF16_prediction':bm['prediction'],'FP32_ties':tie,'BF16_ties':bm['top_ties'],'prediction_changed':sm['prediction']!=bm['prediction'],'FP32_correct':sm['prediction']==g,'BF16_correct':bm['prediction']==g,'FP32_score_meanwrong':float(scores[c]),'BF16_score_meanwrong':float(bf[c])}
 diffs={}
 for c,terms in contrasts.items():
  f=sum(w*scores[k] for k,w in terms.items());bb=sum(w*bf[k] for k,w in terms.items());diffs[c]={'FP32':float(f),'BF16':float(bb),'FP32_sign':sign(f),'BF16_sign':sign(bb),'sign_changed':sign(f)!=sign(bb)}
 rows.append({'id':ident,'metadata_sha256':sha(O/(stem+'.json')),'tensor_sha256':m['tensor_sha256'],'endpoints':endpoints,'production_score_contrasts':diffs,'archive_and_tensor_checks':'PASS','strict_runtime_donor_guard':'Successful frozen source asserts each P/U/UX/UA/UXA K1_Q against N; only N Q9_K1_Q archived, hence no independent perforward raw donor comparison available.'})
out={'status':'PASS','source_sha256':sha(Path(__file__)),'n':8,'source_and_sample_freeze_sha256':sha(O/'SOURCE_AND_SAMPLE_FREEZE.json'),'records':rows,'no_model_or_GPU':True,'primary_scope':'Exact rational arithmetic on stored FP32 production-score meanwrong contrasts; numerical/backend sensitivity, no population inference'}
p=R/'C1_FP32_MATH8_SOL_REVIEW.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'PASS','n':8,'prediction_changes':{c:sum(r['endpoints'][c]['prediction_changed'] for r in rows) for c in conditions},'FP32_ties':{c:sum(len(r['endpoints'][c]['FP32_ties'])>1 for r in rows) for c in conditions},'BF16_ties':{c:sum(len(r['endpoints'][c]['BF16_ties'])>1 for r in rows) for c in conditions},'contrast_sign_changes':{c:sum(r['production_score_contrasts'][c]['sign_changed'] for r in rows) for c in contrasts}}))
