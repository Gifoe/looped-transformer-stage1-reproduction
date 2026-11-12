"""Frozen H3/H4 checks: full validation/test; same candidate BF16 exact rational sensitivity."""
import sys,json,hashlib,gzip,statistics,math
from pathlib import Path
from fractions import Fraction
from safetensors.torch import load_file
ROOT=Path('/home/luansiyu/loop-latent');RUN=ROOT/'orchestration_runs/pqr-round3-20261111';OLD=ROOT/'orchestration_runs/round2-full-20261111';sys.path.insert(0,str(ROOT/'research_plans/pqr_eight_corner_code_20261111'))
from pqr_metrics import NAMES
OUT=ROOT/'research_plans/round3_prospective_val_test_20261111';OUT.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
plan=ROOT/'research_plans/ROUND_3_TRAIN_INFORMED_PROSPECTIVE_CHECKS.md';assert sha(plan)=='57c6274890273dfdd6d1d8831ca9a5ea93f103b05a4a6674b1fef92e7f7916d0'
oldmap={'111':'core_K1C_K1R','112':'core_K1C_K2R','221':'core_K2C_K1R','222':'core_K2C_K2R'};oldnames={'111':'K1C_K1R','112':'K1C_K2R','221':'K2C_K1R','222':'K2C_K2R'}
def measures(v):
 return {'D1':v['121']-v['211'],'D2':v['122']-v['212'],'T1':(v['212']-v['112'])-(v['211']-v['111']),'T2':(v['222']-v['122'])-(v['221']-v['121'])}
def fmt(v):return str(v.numerator)+'/'+str(v.denominator)
def describe(xs):
 y=sorted(xs);n=len(y);return {'n':n,'mean':float(sum(xs)/n) if n else None,'median':float(statistics.median(y)) if n else None,'positive':sum(x>0 for x in y),'negative':sum(x<0 for x in y),'zero':sum(x==0 for x in y),'min':float(y[0]) if n else None,'q25':float(y[(n-1)//4]) if n else None,'q75':float(y[3*(n-1)//4]) if n else None,'max':float(y[-1]) if n else None}
def summary(rows):
 out={'n':len(rows)}
 for mode in ('scores_all','scores_fixed','bf16_all','bf16_fixed'):
  out[mode]={k:describe([Fraction(r[mode][k]) if mode.startswith('bf16') else r[mode][k] for r in rows]) for k in ('D1','D2','T1','T2')}
 out['corners']={k:{'correct':sum(r['corners'][k]['prediction']==r['gold'] for r in rows),'top_ties':sum(r['corners'][k]['top_tie'] for r in rows)} for k in NAMES}
 edges={}
 for bit in (0,1):
  for a in NAMES:
   if a[bit]!='1':continue
   b=a[:bit]+'2'+a[bit+1:];edges[a+'->'+b]={'changed':sum(r['corners'][a]['prediction']!=r['corners'][b]['prediction'] for r in rows),'rescued':sum(r['corners'][a]['prediction']!=r['gold'] and r['corners'][b]['prediction']==r['gold'] for r in rows),'damaged':sum(r['corners'][a]['prediction']==r['gold'] and r['corners'][b]['prediction']!=r['gold'] for r in rows)};edges[a+'->'+b]['net']=edges[a+'->'+b]['rescued']-edges[a+'->'+b]['damaged']
 out['P_Q_edges']=edges;return out
for split,n,expected_hash in [('validation',299,'0a0fce9188afbe18d677121b9fa3f21130b9baf34c1887a9009203cd3134a417'),('test',1172,'c7ced839d0079ea9732aba26c201c4bd3dbf2670c4499993c6e143291f76d951')]:
 cv=RUN/('coverage_'+split+'.json');assert sha(cv)==expected_hash;c=json.loads(cv.read_text());assert c['status']=='PQR_COMPLETE_CUDA_REPLAY_ZERO' and c['complete_questions']==c['expected_questions']==n and c['replayed_condition_logits']==4*n
 oldcv=OLD/('coverage_'+split+'.json');oc=json.loads(oldcv.read_text());old={q['id']:q for q in oc['questions']};assert len(old)==n and set(old)=={q['id'] for q in c['questions']};rows=[]
 for q in c['questions']:
  ident=q['id'];key=hashlib.sha256(ident.encode()).hexdigest();mp=RUN/split/(key+'.json');tp=mp.with_suffix('.safetensors');om=OLD/split/(key+'.json');ot=om.with_suffix('.safetensors')
  assert sha(mp)==q['metadata_sha256'] and sha(tp)==q['logits_sha256'] and sha(om)==old[ident]['metadata_sha256'] and sha(ot)==old[ident]['logits_sha256'];payload=json.loads(mp.read_text())['payload'];oj=json.loads(om.read_text());v=payload['result'];b=oj['payload']['result'];corners=v['corners'];assert payload['id']==ident and payload['binding_sha256']==c['binding_sha256'] and payload['old_metadata_sha256']==sha(om) and set(corners)==set(NAMES)
  for k,name in oldnames.items():assert corners[k]==b['conditions'][name]
  g=v['gold'];wrong=v['fixed_wrong'];targets=v['target_ids'];z=load_file(str(tp));w=load_file(str(ot));scores={k:corners[k]['scores'] for k in NAMES};raw={}
  for k in NAMES:
   t=w[oldmap[k]] if k in oldmap else z['corner_'+k];assert str(t.dtype)=='torch.bfloat16' and list(t.shape)==[1,49152];raw[k]=[Fraction(float(t[0,i])) for i in targets]
  vals=[];exact=[]
  for j in range(len(targets)):
   if j==g:continue
   vals.append((j,measures({k:scores[k][g]-scores[k][j] for k in NAMES})));exact.append((j,measures({k:raw[k][g]-raw[k][j] for k in NAMES})))
  avg=lambda arr,k:sum(v[k] for _,v in arr)/len(arr)
  fixed=dict(vals)[wrong];efixed=dict(exact)[wrong];k1=b['baseline_K1']['prediction']==g;k2=b['baseline_K2']['prediction']==g
  r={'id':ident,'gold':g,'fixed_wrong':wrong,'target_ids':targets,'corners':corners,'raw_candidate_logits':{k:[float(x) for x in raw[k]] for k in NAMES},'scores_all':{k:avg(vals,k) for k in fixed},'scores_fixed':fixed,'bf16_all':{k:fmt(avg(exact,k)) for k in fixed},'bf16_fixed':{k:fmt(efixed[k]) for k in fixed},'group':{(False,True):'rescue',(True,False):'damage',(True,True):'stable_correct',(False,False):'stable_wrong'}[(k1,k2)],'natural_no_top_tie':not b['baseline_K1']['top_tie'] and not b['baseline_K2']['top_tie'],'eight_no_top_tie':all(not corners[k]['top_tie'] for k in NAMES),'pqr_metadata_sha256':sha(mp),'pqr_logits_sha256':sha(tp),'round2_metadata_sha256':sha(om),'round2_logits_sha256':sha(ot)};rows.append(r)
 assert len(rows)==n
 fpath=OUT/(split+'_rows.jsonl.gz')
 with gzip.open(fpath,'xt',encoding='utf8') as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
 report={'status':'TRAIN_INFORMED_PROSPECTIVE_PROJECT_INTERNAL','split':split,'plan_sha256':sha(plan),'coverage_sha256':sha(cv),'old_coverage_sha256':sha(oldcv),'rows_sha256':sha(fpath),'script_sha256':sha(Path(__file__)),'all':summary(rows),'natural_no_top_tie':summary([r for r in rows if r['natural_no_top_tie']]),'eight_no_top_tie':summary([r for r in rows if r['eight_no_top_tie']]),'groups':{g:summary([r for r in rows if r['group']==g]) for g in ('rescue','damage','stable_correct','stable_wrong')}}
 out=OUT/(split+'_summary.json')
 with out.open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
 print('PROSPECTIVE_READY',split,sha(out),sha(fpath),flush=True)
