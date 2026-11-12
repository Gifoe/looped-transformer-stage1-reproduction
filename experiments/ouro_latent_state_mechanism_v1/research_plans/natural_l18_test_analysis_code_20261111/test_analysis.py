"""Test-only L18 descriptive analysis after exact CPU pre-audit."""
import os,json,hashlib,gzip
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from source_input import ROOT,sha
RUN=ROOT/'research_plans/natural_l18_full_run_20261111';COV=ROOT/'research_plans/NATURAL_L18_TEST_PRELIMINARY_COVERAGE_20261111.json';GEO=ROOT/'research_plans/natural_r_geometry_20261111';OUT=ROOT/'research_plans/natural_l18_test_analysis_20261111'
def require(x,msg):
 if not x:raise ValueError(msg)
def desc(v):
 a=np.asarray(v,dtype='float64')
 if not len(a):return {'n':0}
 q=np.quantile(a,[.05,.25,.5,.75,.95]);return {'n':len(a),'mean':float(a.mean()),'median':float(q[2]),'q05':float(q[0]),'q25':float(q[1]),'q75':float(q[3]),'q95':float(q[4]),'positive':int((a>0).sum()),'negative':int((a<0).sum()),'zero':int((a==0).sum())}
def main():
 c=json.loads(COV.read_text());require(c['status']=='CPU_SOURCE_BASELINE_IDENTITY_PASS_INDEPENDENT_CUDA_PENDING' and c['count']==1172 and c['analysis_script_sha256']==sha(Path(__file__)) and c['amendment_sha256']=='03f5807d4616707804234975dac69cc1157bc0ac503b2b14eda653b7c6d54ed5' and c['forecast_sha256']=='6ba60011820c2b2c73c2764af7fb1851b128012606461af11bbdb1fe65bfc31d','coverage');entries={(q['split'],q['id']):q for q in c['questions']};require(len(entries)==1172,'duplicate');require(sha(GEO/'source.json')=='8e22e6ef7d5cd920a7f5ac4faae0d7df31f002a094346e289cb325f9d5443ed2','geometry source SHA');source=json.loads((GEO/'source.json').read_text());p=GEO/'test_rows.jsonl';require(sha(p)==source['test']['rows_sha256'],'geometry SHA');rows=[]
 with p.open() as f:
  for line in f:
   natural=json.loads(line);key=('test',natural['id']);require(key in entries,'missing ID');q=entries[key];name='test_'+hashlib.sha256(natural['id'].encode()).hexdigest();mp=RUN/(name+'.json');tp=mp.with_suffix('.safetensors');require(sha(mp)==q['metadata_sha256'] and sha(tp)==q['tensor_sha256'],'artifact SHA');m=json.loads(mp.read_text());require(m['id']==natural['id'] and m['gold']==natural['gold'] and m['targets']==natural['targets'] and m['fixed_wrong']==natural['fixed_wrong'] and m['scores']['B1']==natural['scores'][0] and m['scores']['B2']==natural['scores'][1],'natural identity')
   scores=m['scores'];gold=m['gold'];p1=m['predictions']['B1'];p2=m['predictions']['B2'];group=('stable_correct' if p2==gold else 'damage') if p1==gold else ('rescue' if p2==gold else 'stable_wrong');require(natural['group']==group and natural['natural_no_top_tie']==(not m['top_ties']['B1'] and not m['top_ties']['B2']),'natural behavior labels');wrong=m['fixed_wrong'];others=[j for j in range(len(m['targets'])) if j!=gold];t=load_file(str(tp));raw={k:[float(z[0,tid]) for tid in m['targets']] for k,z in t.items()}
   def metric(b,t):
    if b not in scores or t not in scores:return None
    per=[(scores[b][gold]-scores[b][j])-(scores[t][gold]-scores[t][j]) for j in others]
    return {'all_wrong':float(np.mean(per)),'fixed_wrong':(scores[b][gold]-scores[b][wrong])-(scores[t][gold]-scores[t][wrong]),'per_wrong':per}
   dynamic={k:v[gold]-max(v[j] for j in others) for k,v in scores.items()}
   rows.append({'id':natural['id'],'group':natural['group'],'demo':natural['demo'],'natural_no_top_tie':natural['natural_no_top_tie'],'gold':gold,'fixed_wrong':wrong,'targets':m['targets'],'target_logits_bf16':raw,'scores':scores,'dynamic_gap':dynamic,'predictions':m['predictions'],'top_ties':m['top_ties'],'controls':m['controls'],'L2':metric('B2','T21'),'G1':metric('T12','B1'),'L2_norm':metric('B2','T21_norm'),'G1_norm':metric('T12_norm','B1')})
 require(len(rows)==1172 and len({r['id'] for r in rows})==1172 and sum(r['demo'] for r in rows)==0,'rows');OUT.mkdir(exist_ok=False)
 rf=OUT/'rows.jsonl.gz'
 with gzip.open(rf,'xt') as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
 groups={'all':rows,'non_demo':[r for r in rows if not r['demo']],'demo':[r for r in rows if r['demo']]}
 for scope in ('all','non_demo'):
  b=groups[scope];groups[scope+'/natural_no_tie']=[r for r in b if r['natural_no_top_tie']]
  for g in ('rescue','damage','stable_correct','stable_wrong'):groups[scope+'/'+g]=[r for r in b if r['group']==g];groups[scope+'/natural_no_tie/'+g]=[r for r in b if r['group']==g and r['natural_no_top_tie']]
 out={}
 for name,rs in groups.items():
  v={}
  for key in ('L2','G1','L2_norm','G1_norm'):
   for ref in ('all_wrong','fixed_wrong'):v[key+'_'+ref]=desc([r[key][ref] for r in rs if r[key] is not None])
   v[key+'_undefined_ids']=[r['id'] for r in rs if r[key] is None]
  for ref in ('all_wrong','fixed_wrong'):
   l=[r['L2'][ref] for r in rs];g=[r['G1'][ref] for r in rs];v['joint_'+ref]={'both_positive':sum(a>0 and b>0 for a,b in zip(l,g)),'both_negative':sum(a<0 and b<0 for a,b in zip(l,g)),'L2_positive_G1_negative':sum(a>0 and b<0 for a,b in zip(l,g)),'L2_negative_G1_positive':sum(a<0 and b>0 for a,b in zip(l,g)),'has_zero':sum(a==0 or b==0 for a,b in zip(l,g))}
   for key in ('L2','G1'):
    pairs=sorted((r[key][ref],r['id']) for r in rs);v[key+'_'+ref+'_extremes']={'min':pairs[:5],'max':pairs[-5:]}
  for key in ('L2','G1'):v[key+'_fixed_vs_all_sign_conflict']=sum(r[key]['all_wrong']*r[key]['fixed_wrong']<0 for r in rs)
  for cond in ('B1','B2','I1','I2','T21','T12','T21_norm','T12_norm'):
   rr=[r for r in rs if cond in r['predictions']];v[cond+'_accuracy']={'n':len(rr),'correct':sum(r['predictions'][cond]==r['gold'] for r in rr),'top_tie':sum(r['top_ties'][cond] for r in rr)}
  for before,after in [('B2','T21'),('B1','T12'),('B2','T21_norm'),('B1','T12_norm')]:
   rr=[r for r in rs if after in r['predictions']];recover=sum(r['predictions'][before]!=r['gold'] and r['predictions'][after]==r['gold'] for r in rr);damage=sum(r['predictions'][before]==r['gold'] and r['predictions'][after]!=r['gold'] for r in rr)
   v[before+'_to_'+after]={'n':len(rr),'unchanged':sum(r['predictions'][before]==r['predictions'][after] for r in rr),'changed':sum(r['predictions'][before]!=r['predictions'][after] for r in rr),'correct_to_wrong':damage,'wrong_to_correct':recover,'wrong_to_different_wrong':sum(r['predictions'][before]!=r['gold'] and r['predictions'][after]!=r['gold'] and r['predictions'][before]!=r['predictions'][after] for r in rr),'net_correct':recover-damage,'net_loss':damage-recover,'top_tie_introduced':sum(not r['top_ties'][before] and r['top_ties'][after] for r in rr),'top_tie_removed':sum(r['top_ties'][before] and not r['top_ties'][after] for r in rr)}
  for main,norm in [('L2','L2_norm'),('G1','G1_norm')]:
   rr=[r for r in rs if r[norm] is not None]
   for ref in ('all_wrong','fixed_wrong'):
    a=[r[main][ref] for r in rr];b=[r[norm][ref] for r in rr];v[main+'_norm_minus_full_'+ref]=desc([y-x for x,y in zip(a,b)]);v[main+'_norm_sign_'+ref]={'same_nonzero':sum(x*y>0 for x,y in zip(a,b)),'opposite':sum(x*y<0 for x,y in zip(a,b)),'has_zero':sum(x==0 or y==0 for x,y in zip(a,b))}
  for effect,full,norm in [('L2','T21','T21_norm'),('G1','T12','T12_norm')]:
   rr=[r for r in rs if norm in r['predictions']];v[effect+'_norm_prediction_diff']=sum(r['predictions'][norm]!=r['predictions'][full] for r in rr);v[effect+'_norm_bf16_error']=desc([r['controls'][norm]['norm_error'] for r in rr])
  v['T12_same_as_B2']=sum(r['predictions']['T12']==r['predictions']['B2'] for r in rs);v['T21_same_as_B1']=sum(r['predictions']['T21']==r['predictions']['B1'] for r in rs);v['natural_baseline_same']=sum(r['predictions']['B1']==r['predictions']['B2'] for r in rs)
  v['rescue_T21_undo']=sum(r['group']=='rescue' and r['predictions']['T21']!=r['gold'] for r in rs);v['rescue_T12_gold']=sum(r['group']=='rescue' and r['predictions']['T12']==r['gold'] for r in rs);v['damage_T21_restore']=sum(r['group']=='damage' and r['predictions']['T21']==r['gold'] for r in rs);v['damage_T12_wrong']=sum(r['group']=='damage' and r['predictions']['T12']!=r['gold'] for r in rs)
  out[name]=v
 result={'status':'TEST_TRAIN_INFORMED_EXPLORATORY_CPU_SOURCE_BASELINE_IDENTITY_PASS_INDEPENDENT_CUDA_PENDING','coverage_sha256':sha(COV),'rows_sha256':sha(rf),'script_sha256':sha(Path(__file__)),'definitions':{'L2':'gap(B2)-gap(T21)','G1':'gap(T12)-gap(B1)','all_wrong':'within-question equally weighted non-gold gaps'},'groups':out}
 with (OUT/'summary.json').open('x') as f:json.dump(result,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print('L18_TEST_ANALYSIS_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
