"""Descriptive analysis of completed train-only branch rows."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/luansiyu/loop-latent/research_plans/natural_branch_train_exploration_20261111')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(values):
 a=np.asarray(values,dtype='float64')
 if len(a)==0:return {'n':0}
 q=np.quantile(a,[.05,.25,.5,.75,.95]);return {'n':len(a),'mean':float(a.mean()),'q05':float(q[0]),'q25':float(q[1]),'median':float(q[2]),'q75':float(q[3]),'q95':float(q[4]),'positive':int((a>0).sum()),'negative':int((a<0).sum()),'zero':int((a==0).sum())}
def main():
 coverage=json.loads((ROOT/'train_coverage_preliminary.json').read_text());assert coverage['status']=='PRELIMINARY_CPU_ANCHORS_PASS_CUDA_REPLAY_PENDING' and coverage['count']==1119
 rows=json.loads((ROOT/'train_rows.json').read_text());assert digest(ROOT/'train_rows.json')==coverage['rows_sha256'] and len(rows)==1119 and len({x['id'] for x in rows})==1119 and {x['id'] for x in rows}=={q['id'] for q in coverage['questions']} and digest(Path(__file__))==coverage['summary_script_sha256']
 groups={'all':rows,'non_demo':[r for r in rows if not r['demo']],'demo':[r for r in rows if r['demo']]}
 for pop in ('all','non_demo'):
  base=groups[pop]
  for group in ('rescue','damage','stable_correct','stable_wrong'):
   groups[pop+'/'+group]=[r for r in base if r['group']==group]
  groups[pop+'/natural_no_tie']=[r for r in base if r['natural_no_top_tie']]
  for group in ('rescue','damage','stable_correct','stable_wrong'):
   groups[pop+'/natural_no_tie/'+group]=[r for r in base if r['natural_no_top_tie'] and r['group']==group]
 metrics={}
 for group,rr in groups.items():
  layers=[]
  for i in range(24):
   item={'layer':i+1,'n':len(rr)}
   for ref in ('all_wrong','fixed_wrong'):
    for branch in ('attention','mlp'):
     for comp in ('delta','numerator','denominator'):
      for which in ('k1','k2','difference'):
       key=branch+'_'+comp+'_'+which+'_'+ref;item[key]=describe([r['layers'][i][branch+'_'+comp][which][ref] for r in rr])
    da=[r['layers'][i]['attention_delta']['difference'][ref] for r in rr];dm=[r['layers'][i]['mlp_delta']['difference'][ref] for r in rr]
    for branch in ('attention','mlp'):
     for part in ('branch','rounding'):
      for u in (0,1):
       item[branch+'_numerator_'+part+'_k'+str(u+1)+'_'+ref]=describe([r['layers'][i][branch+'_numerator_parts'][u][part][ref] for r in rr])
      item[branch+'_numerator_'+part+'_paired_difference_'+ref]=describe([r['layers'][i][branch+'_numerator_parts'][1][part][ref]-r['layers'][i][branch+'_numerator_parts'][0][part][ref] for r in rr])
    item['total_difference_'+ref]=describe([a+b for a,b in zip(da,dm)])
    item['joint_sign_'+ref]={'both_positive':sum(a>0 and b>0 for a,b in zip(da,dm)),'both_negative':sum(a<0 and b<0 for a,b in zip(da,dm)),'opposite':sum(a*b<0 for a,b in zip(da,dm)),'has_zero':sum(a==0 or b==0 for a,b in zip(da,dm))}
    for name,values in [('dA',da),('dM',dm),('dT',[a+b for a,b in zip(da,dm)])]:
     pairs=sorted(zip(values,(r['id'] for r in rr)));item[name+'_extremes_'+ref]={'min':pairs[:3],'max':pairs[-3:]}
   for which in ('attention','mlp','total'):
    def delta(r,ref):
     x=r['layers'][i];a=x['attention_delta']['difference'][ref];b=x['mlp_delta']['difference'][ref];return a if which=='attention' else b if which=='mlp' else a+b
    item[which+'_fixed_vs_all_sign_conflicts']=sum(delta(r,'all_wrong')*delta(r,'fixed_wrong')<0 for r in rr)
   layers.append(item)
  totals={}
  for ref in ('all_wrong','fixed_wrong'):
   for name,fn in [('initial_k1',lambda r:r['telescoping']['initial_k1'][ref]),('initial_k2',lambda r:r['telescoping']['initial_k2'][ref]),('endpoint_k1',lambda r:r['telescoping']['final_k1'][ref]),('endpoint_k2',lambda r:r['telescoping']['final_k2'][ref]),('initial_difference',lambda r:r['telescoping']['initial_k2'][ref]-r['telescoping']['initial_k1'][ref]),('endpoint_difference',lambda r:r['telescoping']['final_k2'][ref]-r['telescoping']['final_k1'][ref]),('sum_dA',lambda r:sum(x['attention_delta']['difference'][ref] for x in r['layers'])),('sum_dM',lambda r:sum(x['mlp_delta']['difference'][ref] for x in r['layers']))]:totals[name+'_'+ref]=describe([fn(r) for r in rr])
  metrics[group]={'layers':layers,'telescope':totals}
 report={'status':'TRAIN_EXPLORATORY_CPU_ANCHORS_PASS_CUDA_REPLAY_PENDING','scope':'Natural trajectory FP64 lens; not causal, no val/test read.','coverage_sha256':digest(ROOT/'train_coverage_preliminary.json'),'rows_sha256':digest(ROOT/'train_rows.json'),'script_sha256':digest(Path(__file__)),'groups':metrics}
 with (ROOT/'train_summary.json').open('x') as f:json.dump(report,f,sort_keys=True,allow_nan=False);f.write('\n')
 print('TRAIN_SUMMARY_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
