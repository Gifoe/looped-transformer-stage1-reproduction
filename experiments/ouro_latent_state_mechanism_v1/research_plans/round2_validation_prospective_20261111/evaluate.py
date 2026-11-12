import json,hashlib,pathlib,gzip,statistics,datetime
P=pathlib.Path('/home/luansiyu/loop-latent');run=P/'orchestration_runs/round2-full-20261111';out=pathlib.Path(__file__).parent;pre=P/'research_plans/TRAIN_INFORMED_PROSPECTIVE_CHECKS.md';expected='725ca0140f56a972bbec24fd48307615d3c0356b07133dbfec4bddd23c4ad756';cov=run/'coverage_validation.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode();assert sha(pre)==expected;c=json.loads(cov.read_text());covhash=sha(cov);assert c['status']=='COMPLETE_CUDA_REPLAY_ZERO_TOLERANCE' and c['expected_questions']==c['complete_questions']==299 and c['replayed_condition_logits']==3289
rows=c['questions'];assert len(rows)==len({q['id'] for q in rows})==299; results=[]
for q in rows:
 assert q['status']=='complete';key=hashlib.sha256(q['id'].encode()).hexdigest();p=run/'validation'/(key+'.json');assert sha(p)==q['metadata_sha256'];j=json.loads(p.read_text());pl=j['payload'];assert j['payload_sha256']==hashlib.sha256(canon(pl)).hexdigest();i=pl['identity'];r=pl['result'];assert i['id']==q['id'] and i['split']=='validation' and pl['logits_sha256']==q['logits_sha256'] and {k:i[k] for k in c['source']}==c['source'];z=r['conditions'];w=r['norm_conditions'];m=lambda x:x['gold_vs_fixed_wrong_gap'];m11=m(z['K1C_K1R']);m12=m(z['K1C_K2R']);m21=m(z['K2C_K1R']);m22=m(z['K2C_K2R']);row={'id':q['id'],'demo':i['is_demonstration'],'natural_no_top_tie':not r['baseline_K1']['top_tie'] and not r['baseline_K2']['top_tie'],'rescue':r['baseline_K1']['prediction']!=r['gold'] and r['baseline_K2']['prediction']==r['gold'],'D':m21-m12,'C_given_R2':m22-m12,'C_given_R1':m21-m11,'R_given_C2':m22-m21,'R_given_C1':m12-m11,'interaction':m22-m12-m21+m11,'Crollback_correct':z['K1C_K2R']['prediction']==r['gold'],'Rrollback_correct':z['K2C_K1R']['prediction']==r['gold']}
 for region,pk in [('C','K1C_K2R'),('R','K2C_K1R')]:
  B=m22;Pval=m(z[pk]);A=m(w[region+'_K1DIR_K2NORM']);N=m(w[region+'_K2DIR_K1NORM']);row[region+'_TA']=abs(A-B)-abs(A-Pval);row[region+'_TN']=abs(N-Pval)-abs(N-B);row[region+'_full_rollback_gap_change']=Pval-B
  gg=r['gold'];cand=[]
  for k in range(len(z[pk]['scores'])):
   if k==gg:continue
   gap=lambda o:o['scores'][gg]-o['scores'][k]
   b,p,a,n=gap(z['K2C_K2R']),gap(z[pk]),gap(w[region+'_K1DIR_K2NORM']),gap(w[region+'_K2DIR_K1NORM']);cand.append((abs(a-b)-abs(a-p),abs(n-p)-abs(n-b)))
  row[region+'_candidate_mean_TA']=statistics.mean(x[0] for x in cand);row[region+'_candidate_mean_TN']=statistics.mean(x[1] for x in cand)
 results.append(row)
assert len(results)==299 and not any(r['demo'] for r in results) and sha(cov)==covhash and sha(pre)==expected
def stats(x):return {'n':len(x),'mean':statistics.mean(x) if x else None,'median':statistics.median(x) if x else None,'positive':sum(y>0 for y in x),'negative':sum(y<0 for y in x),'zero':sum(y==0 for y in x)}
def summarize(rs):
 metrics=('D','C_given_R2','C_given_R1','R_given_C2','R_given_C1','interaction','C_TA','C_TN','R_TA','R_TN','C_full_rollback_gap_change','R_full_rollback_gap_change','C_candidate_mean_TA','C_candidate_mean_TN','R_candidate_mean_TA','R_candidate_mean_TN');res=[r for r in rs if r['rescue']];return {'n':len(rs),'metrics':{k:stats([r[k] for r in rs]) for k in metrics},'rescue_n':len(res),'rescue_grid':{str(int(a))+str(int(b)):sum(r['Crollback_correct']==a and r['Rrollback_correct']==b for r in res) for a in (False,True) for b in (False,True)}}
report={'status':'VALIDATION_TRAIN_INFORMED_EXPLORATORY_NOT_INDEPENDENT','scope':'validation complete coverage only; test remains blinded; no post hoc threshold','pre_sha256':expected,'coverage_sha256':covhash,'source':c['source'],'all':summarize(results),'non_demo':summarize([r for r in results if not r['demo']]),'natural_no_top_tie':summarize([r for r in results if r['natural_no_top_tie']]),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with gzip.open(out/'validation_question_metrics.jsonl.gz','xb') as f:
 for r in results:f.write(canon(r)+b'\n')
report['question_metrics_sha256']=sha(out/'validation_question_metrics.jsonl.gz')
with (out/'validation_summary.json').open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
print('COMPLETE coverage',covhash,'questions',len(results),'summary_sha',sha(out/'validation_summary.json'),'rows_sha',report['question_metrics_sha256'])
