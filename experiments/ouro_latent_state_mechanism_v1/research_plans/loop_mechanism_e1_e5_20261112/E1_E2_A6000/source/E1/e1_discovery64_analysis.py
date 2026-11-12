"""Analyze complete independently audited pre-effect stratified discovery64 only."""
import json,gzip,hashlib,os,datetime
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
from safetensors.torch import load_file
R=Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E1';RUN=E/'train';OLD=R/'research_plans/natural_qkv_eight_corner_20261112';A=E/'discovery64_cuda_audit.json';SEL=B/'orders/E1_DISCOVERY64_FROZEN.json';MODEL=R/'model/model.safetensors'; OUT=E/'discovery64_analysis'
if not __debug__:raise RuntimeError('optimized Python forbidden')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def mean(x):return float(np.mean(x))
cfgpath=R/'model/config.json';config_sha=sha(cfgpath);assert config_sha=='ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3';selection_sha=sha(SEL);script_sha=sha(Path(__file__));a=json.loads(A.read_text());sel=json.loads(SEL.read_text());assert a['status']=='DISCOVERY64_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED' and a['count']==64 and a['selection_sha256']==sha(SEL);expected={v['id']:g for g,vs in sel['groups'].items() for v in vs};assert len(expected)==64 and {x['id'] for x in a['questions']}==set(expected);prior_sha=sha(A);model_sha=sha(MODEL);assert model_sha=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af';cfg=json.loads((R/'model/config.json').read_text());eps=cfg['rms_norm_eps']
with safe_open(str(MODEL),framework='pt',device='cpu') as f:W=f.get_tensor('lm_head.weight').float().numpy().astype('float64');gamma=f.get_tensor('model.norm.weight').float().numpy().astype('float64')
rows=[]
for q in a['questions']:
 ident=q['id'];key=hashlib.sha256(ident.encode()).hexdigest();mp=RUN/(key+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==q['metadata_sha256'] and sha(tp)==q['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));op=OLD/'train'/(key+'.json');assert sha(op)==q['old_metadata_sha256'];old=json.loads(op.read_text());gold=old['gold'];targets=old['targets'];wrong=[j for j in range(len(targets)) if j!=gold];fixed=old['fixed_wrong'];Wg=(W[targets[gold]]-np.mean(W[[targets[j] for j in wrong]],axis=0));qresults={}
 def fv(scores):return np.array([scores[gold]-scores[j] for j in wrong],dtype='float64')
 def vec(x):return x.float().numpy().astype('float64').ravel()
 def lens(x,stage):
  x=vec(x)
  if stage in ('oproj','attention_localnorm','mlp_raw','mlp_localnorm'):z=Wg@(gamma*x)
  elif stage=='loop_postnorm':z=Wg@x
  else:z=Wg@(gamma*x/np.sqrt(np.mean(x*x)+eps))
  return float(z)
 for qi in (1,2):
  cs=m['controls']['conditions'][str(qi)];prefix=f'q{qi}_';base=old['scores'][f'G{qi}22'];ad=cs['ADD']['scores'];s11=old['scores'][f'G{qi}11'];s12=old['scores'][f'G{qi}12'];s21=old['scores'][f'G{qi}21'];terminal=fv(base)-fv(s21)-fv(s12)+fv(s11);internal=fv(base)-fv(ad);suffix=fv(ad)-fv(s21)-fv(s12)+fv(s11);assert np.allclose(terminal,internal+suffix,rtol=0,atol=1e-10)
  fixed_index=wrong.index(fixed);dyn=lambda sc:float(sc[gold]-max(sc[j] for j in wrong));Idyn=dyn(base)-dyn(s21)-dyn(s12)+dyn(s11);Rdyn=dyn(base)-dyn(ad);Sdyn=dyn(ad)-dyn(s21)-dyn(s12)+dyn(s11);assert abs(Idyn-Rdyn-Sdyn)<1e-10;result={'I_terminal_fixed':float(terminal[fixed_index]),'I_removed_fixed':float(internal[fixed_index]),'I_suffix_fixed':float(suffix[fixed_index]),'I_terminal_dynamic':Idyn,'I_removed_dynamic':Rdyn,'I_suffix_dynamic':Sdyn,'I_terminal_allwrong':mean(terminal),'I_removed_allwrong':mean(internal),'I_suffix_allwrong':mean(suffix),'candidate_I_terminal':terminal.tolist(),'candidate_removed':internal.tolist(),'candidate_suffix':suffix.tolist(),'accounting_max_abs':float(np.max(np.abs(terminal-internal-suffix))),'fixed_wrong':fixed,'donor':cs['donor'],'conditions':{}}
  for condition in ('ID','ADD','ADD_NORM','UNRELATED'):
   if condition not in cs:continue
   sc=cs[condition]['scores'];delta=fv(base)-fv(sc);idx=max(range(len(sc)),key=lambda j:sc[j]);result['conditions'][condition]={'mean_loss_vs_ID':mean(delta),'fixed_loss_vs_ID':float((base[gold]-base[fixed])-(sc[gold]-sc[fixed])),'candidate_loss_vs_ID':delta.tolist(),'dynamic_loss_vs_ID':dyn(base)-dyn(sc),'prediction':idx,'top_tie':sum(v==max(sc) for v in sc)>1,'correct':idx==gold,'scores':sc}
  result['stages']={}
  stages=['oproj','attention_localnorm','residual_after_attention','mlp_raw','mlp_localnorm']+[f'layer{i:02d}_output' for i in range(18,25)]+['loop_postnorm']
  for stage in stages:
   names=[f'old_G{qi}11_{stage}',f'old_G{qi}12_{stage}',f'old_G{qi}21_{stage}',f'q{qi}_ID_{stage}'];vs=[t[n] for n in names];vv=[vec(x) for x in vs];I=vv[3]-vv[2]-vv[1]+vv[0];ls=[lens(x,stage) for x in vs];result['stages'][stage]={'interaction_l2':float(np.linalg.norm(I)),'diagnostic_interaction':float(ls[3]-ls[2]-ls[1]+ls[0]),'diagnostic_kind':'linear_Wgamma_branch_projection' if stage in ('oproj','attention_localnorm','mlp_raw','mlp_localnorm') else ('direct_head_on_postnorm' if stage=='loop_postnorm' else 'fixed_final_RMS_head_lens')}
  qresults[str(qi)]=result
 rows.append({'id':ident,'natural_group':expected[ident],'split':'train','gold':gold,'targets':targets,'q':qresults,'metadata_sha256':q['metadata_sha256'],'tensor_sha256':q['tensor_sha256']})
assert len(rows)==64 and sha(A)==prior_sha and sha(MODEL)==model_sha and sha(cfgpath)==config_sha and sha(SEL)==selection_sha and sha(Path(__file__))==script_sha;OUT.mkdir(exist_ok=False);path=OUT/'rows.jsonl.gz'
with gzip.open(path,'xt') as f:
 for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
summary={'status':'DISCOVERY64_STRATIFIED_EXPLORATORY_ANALYSIS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':64,'scope':'16 per natural group; not representative train mean; prior ARC outcomes exposed; no causal claim for natural K2 necessity. ADD compares mixed SDPA outputs and common K1 receiver suffix.','audit_sha256':sha(A),'selection_sha256':sha(SEL),'rows_sha256':sha(path),'script_sha256':script_sha,'config_sha256':config_sha,'rms_norm_eps':eps,'groups':{}}
for g in ('rescue','damage','stable_correct','stable_wrong'):
 rr=[r for r in rows if r['natural_group']==g];assert len(rr)==16;summary['groups'][g]={}
 for qi in ('1','2'):
  x=[r['q'][qi] for r in rr];summary['groups'][g][qi]={'N':len(x),'mean_terminal_I':mean([z['I_terminal_allwrong'] for z in x]),'mean_removed':mean([z['I_removed_allwrong'] for z in x]),'mean_suffix':mean([z['I_suffix_allwrong'] for z in x]),'mean_unrelated_loss':mean([z['conditions']['UNRELATED']['mean_loss_vs_ID'] for z in x]),'mean_norm_loss':(mean([z['conditions']['ADD_NORM']['mean_loss_vs_ID'] for z in x if 'ADD_NORM' in z['conditions']]) if any('ADD_NORM' in z['conditions'] for z in x) else None),'norm_defined_N':sum('ADD_NORM' in z['conditions'] for z in x),'positive_removed':sum(z['I_removed_allwrong']>0 for z in x),'negative_removed':sum(z['I_removed_allwrong']<0 for z in x),'stage_mean_interaction_l2':{k:mean([z['stages'][k]['interaction_l2'] for z in x]) for k in x[0]['stages']},'stage_mean_diagnostic_interaction':{k:mean([z['stages'][k]['diagnostic_interaction'] for z in x]) for k in x[0]['stages']}}
  detail=summary['groups'][g][qi];detail['paired_controls']={}
  for ctrl in ('UNRELATED','ADD_NORM'):
   pairs=[(r['id'],r['q'][qi]['I_removed_allwrong']-r['q'][qi]['conditions'][ctrl]['mean_loss_vs_ID']) for r in rr if ctrl in r['q'][qi]['conditions']];v=[v for _,v in pairs];detail['paired_controls'][ctrl]={'defined_N':len(v),'median':float(np.median(v)) if v else None,'positive_ids':sorted(i for i,v in pairs if v>0),'negative_ids':sorted(i for i,v in pairs if v<0),'zero_ids':sorted(i for i,v in pairs if v==0)}
  detail['prediction_changes']={}
  for ctrl in ('ADD','ADD_NORM','UNRELATED'):
   pairs=[r['q'][qi]['conditions'] for r in rr if ctrl in r['q'][qi]['conditions']];detail['prediction_changes'][ctrl]={'defined_N':len(pairs),'rescue':sum(not z['ID']['correct'] and z[ctrl]['correct'] for z in pairs),'damage':sum(z['ID']['correct'] and not z[ctrl]['correct'] for z in pairs),'changed':sum(z['ID']['prediction']!=z[ctrl]['prediction'] for z in pairs)}
p=OUT/'summary.json'
with p.open('x') as f:json.dump(summary,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
print('E1_DISCOVERY64_ANALYSIS_COMPLETE',sha(p),flush=True)
