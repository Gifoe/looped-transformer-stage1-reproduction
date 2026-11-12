"""Evaluate pre-frozen train-informed P1-P3 and secondary signs on one fully audited split."""
import gzip,json,sys,hashlib,statistics
from pathlib import Path
B=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_eight_corner_20261112'); S=sys.argv[1]; assert S in ('validation','test'); H='59438c1694501f7bf5dba9d6547fe037258a9bf57bba58aa6ff3cbb21956a7d7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(v):
 if not v:raise ValueError('coverage or forecast mismatch')
p=B/'ROUND1_TRAIN_INFORMED_PROSPECTIVE_CHECKS.md'; c=B/(S+'_independent_cuda_coverage.json'); sm=B/(S+'_analysis/summary.json'); rows=B/(S+'_analysis/rows.jsonl.gz'); x=json.loads(sm.read_text());cov=json.loads(c.read_text());need(sha(p)==H==x['forecast_sha256']==cov['forecast_sha256'] and x['coverage_sha256']==sha(c) and x['rows_sha256']==sha(rows) and x['count']=={'validation':299,'test':1172}[S]);
with gzip.open(rows,'rt',encoding='utf8') as f:r=[json.loads(x) for x in f]
need(cov['status']=='SPLIT_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED' and x['status']=='SPLIT_EIGHT_CORNER_ANALYSIS_COMPLETE' and cov['split']==S==x['split'] and cov['count']==x['count'] and len(r)==x['count'] and {v['id'] for v in r}=={v['id'] for v in cov['questions']} and len({v['id'] for v in r})==len(r))
def group(rr):
 n=len(rr); inter={}
 for i in (1,2):
  k='I_Q'+str(i); d=[z['gap_differences'][k] for z in rr]; pos=sum(z['fraction_signs'][k]>0 for z in rr);inter[k]={'mean':statistics.mean(d),'median':statistics.median(d),'fraction_positive':pos,'N':n,'mean_pass':statistics.mean(d)>0,'median_pass':statistics.median(d)>0,'fraction_majority_pass':pos>n/2}
 edges={}
 for i in (1,2):
  for label,corner in [('K','G'+str(i)+'21'),('V','G'+str(i)+'12')]:
   baseline='G'+str(i)+'11';ds=[z['score_gap'][corner]-z['score_gap'][baseline] for z in rr];res=sum(z['predictions'][baseline]!=z['gold'] and z['predictions'][corner]==z['gold'] for z in rr);dam=sum(z['predictions'][baseline]==z['gold'] and z['predictions'][corner]!=z['gold'] for z in rr);edges[f'{label}_Q{i}']={'mean':statistics.mean(ds),'gap_pass':statistics.mean(ds)>0,'rescue':res,'damage':dam,'net':res-dam,'net_pass':res>dam}
 correct={k:sum(z['predictions'][k]==z['gold'] for z in rr) for k in ['G111','G112','G121','G122','G211','G212','G221','G222']};p3={}
 for i in (1,2):
  x1,x2,x3=f'G{i}12',f'G{i}21',f'G{i}22';p3[str(i)]={'joint_gt_K':correct[x3]>correct[x2],'joint_gt_V':correct[x3]>correct[x1],'K_gt_V':correct[x2]>correct[x1]}
 h1=[z['score_gap']['G121']-z['score_gap']['G112'] for z in rr];h2=[z['score_gap']['G221']-z['score_gap']['G212'] for z in rr];j=[z['gap_differences']['J_QKV'] for z in rr]
 return {'N':n,'P1':inter,'P2':edges,'P3':p3,'correct':correct,'secondary':{'H1_mean':statistics.mean(h1),'H1_positive':statistics.mean(h1)>0,'H2_mean':statistics.mean(h2),'H2_negative':statistics.mean(h2)<0,'J_mean':statistics.mean(j),'J_positive':statistics.mean(j)>0}}
out={'split':S,'forecast_sha256':H,'summary_sha256':sha(sm),'coverage_sha256':sha(c),'rows_sha256':sha(rows),'script_sha256':sha(Path(__file__)),'groups':{}}
for name,rr in [('all',r),('natural_no_tie',[z for z in r if z['natural_no_top_tie']])]+[(g,[z for z in r if z['group']==g]) for g in ('rescue','damage','stable_correct','stable_wrong')]:
 if rr:
  z=group(rr)
  if name not in ('all','natural_no_tie'):
   z['scope']='descriptive natural group; P1/P2/P3 not registered for this group'
   for v in z['P1'].values():
    for key in ('mean_pass','median_pass','fraction_majority_pass'):v.pop(key)
   for v in z['P2'].values():
    for key in ('gap_pass','net_pass'):v.pop(key)
   z.pop('P3')
   sec=z['secondary'];sec.pop('H1_positive');sec.pop('H2_negative');sec.pop('J_positive')
   if name=='rescue':sec['registered_direction_pass']={'H1':sec['H1_mean']>0,'H2':sec['H2_mean']>0,'J':sec['J_mean']>0}
   if name=='stable_correct':sec['registered_direction_pass']={'H1':sec['H1_mean']<0,'H2':sec['H2_mean']<0,'J':sec['J_mean']<0}
  out['groups'][name]=z
target=B/(S+'_analysis/prospective_checks.json');
with target.open('x',encoding='utf8') as f:json.dump(out,f,sort_keys=True,allow_nan=False);f.write('\n')
print('PROSPECTIVE_CHECKS_COMPLETE',S,sha(target),flush=True)
