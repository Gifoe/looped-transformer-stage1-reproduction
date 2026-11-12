import json,gzip,hashlib,pathlib,statistics,datetime,os,math
T=pathlib.Path(os.environ['TEMP']);base={'train':T/'ouro-round2-train-disclosure-20261111','validation':T/'ouro-round2-validation-prospective-20261111'};out=T/'ouro-all-candidate-H2-20261111';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();expected={'train':'cdf479811108ad63c017a694440d81e0e9b0866e30f9938245c87b0884ce5624','validation':'7cae552b7fac1c1832247ac2a784314405c8526a83fb8729e7e5fa6f373bd8f5'}
def stats(values):
 return {'n':len(values),'mean':statistics.mean(values) if values else None,'median':statistics.median(values) if values else None,'positive':sum(x>0 for x in values),'negative':sum(x<0 for x in values),'zero':sum(x==0 for x in values)}
def endpoint(x,p,b):
 if x==p==b:return 'both'
 if x==p:return 'P_only'
 if x==b:return 'B_only'
 return 'third'
def summary(rows):
 metrics=('C_TA_candidate','C_TN_candidate','R_TA_candidate','R_TN_candidate','C_TA_fixed','C_TN_fixed','R_TA_fixed','R_TN_fixed');o={'n':len(rows),'metrics':{k:stats([r[k] for r in rows]) for k in metrics},'endpoints':{},'accuracy_transition':{}}
 for reg in ('C','R'):
  for mode in ('A','N'):
   key=reg+'_'+mode;o['endpoints'][key]={x:sum(r[key+'_endpoint']==x for r in rows) for x in ('both','P_only','B_only','third')};o['accuracy_transition'][key]={x:sum(r[key+'_accuracy_transition']==x for r in rows) for x in ('rescue','damage','stable_correct','stable_wrong')}
 return o
rowsout=[];result={'scope':'post-disclosure exploratory sensitivity; per-question all-wrong-candidate equal weight; not a newly preregistered statistic or independent validation','formulas':{'TA':'abs(A-B)-abs(A-P)','TN':'abs(N-P)-abs(N-B)','gap':'gold minus each non-gold candidate'},'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'splits':{}}
for split in ('train','validation'):
 d=base[split];f=d/('train_scores_compact.jsonl.gz' if split=='train' else 'validation_scores_compact.jsonl.gz');cv=d/('coverage_train.json' if split=='train' else 'coverage_validation.json');assert sha(f)==expected[split];coverage=json.loads(cv.read_text());assert coverage['status']=='COMPLETE_CUDA_REPLAY_ZERO_TOLERANCE';ledger={q['id']:q for q in coverage['questions']};rows=[]
 for line in gzip.open(f,'rt',encoding='utf8'):
  j=json.loads(line);q=ledger[j['id']];assert j['metadata_sha256']==q['metadata_sha256'] and j['logits_sha256']==q['logits_sha256'];g=j['gold'];wrong=j['fixed_wrong'];b=j['conditions']['K2C_K2R'];nopt=len(b['scores']);assert 0<=g<nopt and 0<=wrong<nopt and wrong!=g
  k1=j['baseline_K1'];k2=j['baseline_K2'];good1=k1['prediction']==g;good2=k2['prediction']==g;group={(False,True):'rescue',(True,False):'damage',(True,True):'stable_correct',(False,False):'stable_wrong'}[(good1,good2)];row={'id':j['id'],'split':split,'metadata_sha256':j['metadata_sha256'],'demo':j['is_demonstration'] if split=='train' else j['demo'],'natural_group':group,'natural_no_top_tie':not k1['top_tie'] and not k2['top_tie'],'gold':g,'fixed_wrong':wrong,'candidate_values':{}}
  for reg,pname in [('C','K1C_K2R'),('R','K2C_K1R')]:
   p=j['conditions'][pname];a=j['norm_conditions'][reg+'_K1DIR_K2NORM'];n=j['norm_conditions'][reg+'_K2DIR_K1NORM'];assert all(len(x['scores'])==nopt for x in (p,a,n))
   values={}
   for k in range(nopt):
    if k==g:continue
    gap=lambda x:x['scores'][g]-x['scores'][k];bb,pp,aa,nn=map(gap,(b,p,a,n));values[str(k)]={'TA':abs(aa-bb)-abs(aa-pp),'TN':abs(nn-pp)-abs(nn-bb)}
   row['candidate_values'][reg]=values
   for name in ('TA','TN'):
    row[reg+'_'+name+'_candidate']=statistics.mean(v[name] for v in values.values());row[reg+'_'+name+'_fixed']=values[str(wrong)][name]
   for mode,v in [('A',a),('N',n)]:
    key=reg+'_'+mode;row[key+'_endpoint']=endpoint(v['prediction'],p['prediction'],b['prediction']);isgood=v['prediction']==g;basegood=b['prediction']==g;row[key+'_accuracy_transition']={(False,True):'rescue',(True,False):'damage',(True,True):'stable_correct',(False,False):'stable_wrong'}[(basegood,isgood)]
  rows.append(row)
 assert len(rows)==len(ledger)==len({r['id'] for r in rows})==({'train':1119,'validation':299}[split]);rowsout+=rows;subsets={'all':rows,'non_demo':[r for r in rows if not r['demo']],'natural_no_top_tie':[r for r in rows if r['natural_no_top_tie']]};subsets.update({g:[r for r in rows if r['natural_group']==g] for g in ('rescue','damage','stable_correct','stable_wrong')});result['splits'][split]={'input_sha256':sha(f),'coverage_sha256':sha(cv),'groups':{k:summary(v) for k,v in subsets.items()}}
p=out/'question_metrics.jsonl';p.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for r in rowsout),encoding='utf8');result['rows_sha256']=sha(p);summary_path=out/'summary.json';summary_path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8');print('ROWS_SHA',sha(p),'SUMMARY_SHA',sha(summary_path),'CODE_SHA',sha(pathlib.Path(__file__)));
for split,group in [('train','non_demo'),('validation','all')]:
 z=result['splits'][split]['groups'][group];print(split,group,'N',z['n'],'MEANS',{k:round(v['mean'],6) for k,v in z['metrics'].items()});print('ENDPOINTS',z['endpoints']);print('TRANSITIONS',z['accuracy_transition'])
