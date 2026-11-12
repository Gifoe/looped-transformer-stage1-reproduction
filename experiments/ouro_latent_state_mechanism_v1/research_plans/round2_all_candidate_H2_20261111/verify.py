import json,gzip,hashlib,pathlib,statistics,os
T=pathlib.Path(os.environ['TEMP']);o=T/'ouro-all-candidate-H2-20261111';records=[json.loads(x) for x in (o/'question_metrics.jsonl').read_text(encoding='utf8').splitlines()];by={(r['split'],r['id']):r for r in records};assert len(by)==len(records)==1418
for split,d,name in [('train','ouro-round2-train-disclosure-20261111','train_scores_compact.jsonl.gz'),('validation','ouro-round2-validation-prospective-20261111','validation_scores_compact.jsonl.gz')]:
 for line in gzip.open(T/d/name,'rt',encoding='utf8'):
  j=json.loads(line);r=by.pop((split,j['id']));g=j['gold'];wrong=j['fixed_wrong'];B=j['conditions']['K2C_K2R']
  for region,pname in [('C','K1C_K2R'),('R','K2C_K1R')]:
   P=j['conditions'][pname];A=j['norm_conditions'][region+'_K1DIR_K2NORM'];N=j['norm_conditions'][region+'_K2DIR_K1NORM'];vv={}
   for k in range(len(B['scores'])):
    if k==g:continue
    gap=lambda x:x['scores'][g]-x['scores'][k];b,p,a,n=map(gap,(B,P,A,N));vv[str(k)]={'TA':abs(a-b)-abs(a-p),'TN':abs(n-p)-abs(n-b)}
   assert r['candidate_values'][region]==vv
   for metric in ('TA','TN'):assert r[region+'_'+metric+'_candidate']==statistics.mean(v[metric] for v in vv.values()) and r[region+'_'+metric+'_fixed']==vv[str(wrong)][metric]
   for label,v in [('A',A),('N',N)]:
    x=v['prediction'];p=P['prediction'];b=B['prediction'];ep='both' if x==p==b else 'P_only' if x==p else 'B_only' if x==b else 'third';assert r[region+'_'+label+'_endpoint']==ep
assert not by
print('ALL_1418_H2_CANDIDATE_AND_ENDPOINT_VALUES_INDEPENDENTLY_RECOMPUTED')
