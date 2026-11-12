"""Train-only exact two-factor decomposition; no new forwards."""
import os,json,hashlib,math
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
from safetensors.torch import load_file
from source_input import ROOT,sha,initial_sources
BASE=ROOT/'research_plans/natural_branch_train_exploration_20261111';BRANCH=ROOT/'research_plans/natural_branch_full_run_20261111';PLAN=ROOT/'research_plans/NATURAL_BRANCH_TRAIN_NUMERATOR_SCALE_PLAN_20261111.md';DEST=ROOT/'research_plans/natural_branch_train_numerator_scale_20261111'
def req(x,msg):
 if not x:raise ValueError(msg)
def describe(x):
 a=np.asarray(x,dtype='float64')
 if not len(a):return {'n':0}
 q=np.quantile(a,[.05,.25,.5,.75,.95]);return {'n':len(a),'mean':float(a.mean()),'median':float(q[2]),'q05':float(q[0]),'q25':float(q[1]),'q75':float(q[3]),'q95':float(q[4]),'positive':int((a>0).sum()),'negative':int((a<0).sum()),'zero':int((a==0).sum())}
def main():
 req(sha(PLAN)=='547a950416909958cc9154f18126f5a64e2b46b69f6b9dc554d076f8d4130a0d','plan')
 cov=BASE/'train_coverage_preliminary.json';rf=BASE/'train_rows.json';req(sha(cov)=='61132388ee5403a042cc1ab8fbca17c59192e168866ea6cbac084a49c368440f' and sha(rf)=='c280f4ee3719c96d04e18742050b012e36a7361c197e3affd702c5ddc84bf4af','sources')
 rows=json.loads(rf.read_text());cover=json.loads(cov.read_text());qmap={q['id']:q for q in cover['questions']};req(len(rows)==1119 and len(qmap)==1119 and {r['id'] for r in rows}==set(qmap),'ids');DEST.mkdir(exist_ok=False)
 src=initial_sources();eps=json.loads(src['config'].read_text())['rms_norm_eps']
 with safe_open(str(src['model']),framework='pt',device='cpu') as f:W=f.get_tensor('lm_head.weight').float().numpy().astype('float64');gamma=f.get_tensor('model.norm.weight').float().numpy().astype('float64')
 result=[];worst=0.
 for idx,r in enumerate(rows,1):
  key='train_'+hashlib.sha256(r['id'].encode()).hexdigest();mp=BRANCH/(key+'.json');tp=BRANCH/(key+'.safetensors');q=qmap[r['id']];req(sha(mp)==q['branch_metadata_sha256'] and sha(tp)==q['branch_tensor_sha256'],'branch SHA');t=load_file(str(tp));weights=W[r['targets']];gold=r['gold'];wrong=r['fixed_wrong'];other=[j for j in range(len(r['targets'])) if j!=gold]
  def gap(z,ref):return float(z[gold]-z[wrong]) if ref=='fixed_wrong' else float(np.mean([z[gold]-z[j] for j in other]))
  layers=[]
  for i in range(24):
   it={'layer':i+1}
   for branch,left,right,vname in [('attention','x','y','a'),('mlp','y','z','b')]:
    f=[];n={ref:[] for ref in ('all_wrong','fixed_wrong')}
    for u in (0,1):
     vals={k:t[f'u{u}_layer{i:02d}_{k}'].float().numpy()[0].astype('float64') for k in (left,right,vname)}
     f.append(.5*(1/math.sqrt(np.mean(vals[left]**2)+eps)+1/math.sqrt(np.mean(vals[right]**2)+eps)))
     z=weights@(gamma*vals[vname])
     for ref in n:n[ref].append(gap(z,ref))
    it[branch]={}
    for ref in n:
     n1,n2=n[ref];f1,f2=f;proj=.5*(f1+f2)*(n2-n1);scale=.5*(n1+n2)*(f2-f1);diff=f2*n2-f1*n1;error=abs(proj+scale-diff);worst=max(worst,error)
     req(np.isclose(proj+scale,diff,rtol=1e-10,atol=1e-10),'two-factor identity');p=r['layers'][i][branch+'_numerator_parts'];req(np.isclose(f1*n1,p[0]['branch'][ref],rtol=1e-10,atol=1e-10) and np.isclose(f2*n2,p[1]['branch'][ref],rtol=1e-10,atol=1e-10),'source consistency')
     it[branch][ref]={'n1':n1,'n2':n2,'f1':f1,'f2':f2,'projection_term':proj,'scale_term':scale,'weighted_difference':diff}
   layers.append(it)
  result.append({'id':r['id'],'demo':r['demo'],'group':r['group'],'natural_no_top_tie':r['natural_no_top_tie'],'layers':layers})
  if idx%100==0:print('SCALE_AUDITED',idx,flush=True)
 def write(n,obj):
  with (DEST/n).open('x') as f:json.dump(obj,f,sort_keys=True,allow_nan=False);f.write('\n')
 write('rows.json',result)
 groups={'all':result,'non_demo':[r for r in result if not r['demo']],'demo':[r for r in result if r['demo']]}
 for scope in ('all','non_demo'):
  rr=groups[scope]
  groups[scope+'/natural_no_tie']=[r for r in rr if r['natural_no_top_tie']]
  for g in ('rescue','damage','stable_correct','stable_wrong'):groups[scope+'/'+g]=[r for r in rr if r['group']==g];groups[scope+'/natural_no_tie/'+g]=[r for r in rr if r['group']==g and r['natural_no_top_tie']]
 summary={}
 for group,rr in groups.items():
  ls=[]
  for i in range(24):
   d={'layer':i+1}
   for branch in ('attention','mlp'):
    for ref in ('all_wrong','fixed_wrong'):
     for k in ('n1','n2','f1','f2','projection_term','scale_term','weighted_difference'):d[branch+'_'+ref+'_'+k]=describe([r['layers'][i][branch][ref][k] for r in rr])
   ls.append(d)
  summary[group]=ls
 write('summary.json',{'status':'TRAIN_DESCRIPTIVE_SCALE_CPU_ONLY_CUDA_REPLAY_PENDING','rows_sha256':sha(DEST/'rows.json'),'coverage_sha256':sha(cov),'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),'max_identity_error':worst,'groups':summary})
 print('TRAIN_SCALE_COMPLETE',len(result),'max_error',worst,flush=True)
if __name__=='__main__':main()
