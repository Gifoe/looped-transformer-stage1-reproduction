"""Train-only disclosed ROUND3 descriptive analysis, no val/test effects."""
import sys,json,hashlib,gzip,os,statistics
from pathlib import Path
ROOT=Path('/home/luansiyu/loop-latent');RUN=ROOT/'orchestration_runs/pqr-round3-20261111';OLD=ROOT/'orchestration_runs/round2-full-20261111';sys.path.insert(0,str(ROOT/'research_plans/pqr_eight_corner_code_20261111'))
from pqr_metrics import evaluate,describe,NAMES
OUT=ROOT/'research_plans/round3_train_exploratory_20261111';OUT.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
cv=RUN/'coverage_train.json';assert sha(cv)=='93654dd3a4a53312029e2b33f37253043b5c3086a9330e70571fc47c259dd72d';c=json.loads(cv.read_text());assert c['status']=='PQR_COMPLETE_CUDA_REPLAY_ZERO' and c['expected_questions']==c['complete_questions']==1119 and c['replayed_condition_logits']==4476
oldcov=json.loads((OLD/'coverage_train.json').read_text());old={q['id']:q for q in oldcov['questions']};assert len(old)==1119 and {q['id'] for q in c['questions']}==set(old)
oldnames={'111':'K1C_K1R','112':'K1C_K2R','221':'K2C_K1R','222':'K2C_K2R'};items=[]
for q in c['questions']:
 ident=q['id'];key=hashlib.sha256(ident.encode()).hexdigest();mp=RUN/'train'/(key+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==q['metadata_sha256'] and sha(tp)==q['logits_sha256'];j=json.loads(mp.read_text());payload=j['payload'];canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode();assert hashlib.sha256(canonical).hexdigest()==j['payload_sha256'] and payload['id']==ident and payload['split']=='train' and payload['binding_sha256']==c['binding_sha256']
 om=OLD/'train'/(key+'.json');assert sha(om)==old[ident]['metadata_sha256'] and sha(om)==payload['old_metadata_sha256'];oj=json.loads(om.read_text());v=payload['result'];b=oj['payload']['result'];corners=v['corners'];assert set(corners)==set(NAMES)
 for k,n in oldnames.items():assert corners[k]==b['conditions'][n]
 g=v['gold'];wrong=v['fixed_wrong'];scores={k:corners[k]['scores'] for k in NAMES};assert all(len(s)==len(v['target_ids']) and s[g]-s[wrong]==corners[k]['gold_vs_fixed_wrong_gap'] for k,s in scores.items())
 k1=b['baseline_K1']['prediction']==g;k2=b['baseline_K2']['prediction']==g;group={(False,True):'rescue',(True,False):'damage',(True,True):'stable_correct',(False,False):'stable_wrong'}[(k1,k2)]
 vals={k:scores[k][g]-scores[k][wrong] for k in NAMES};metrics=evaluate(vals);allc={str(i):evaluate({k:scores[k][g]-scores[k][i] for k in NAMES}) for i in range(len(scores['111'])) if i!=g}
 items.append({'id':ident,'demo':oj['payload']['identity']['is_demonstration'],'natural_group':group,'natural_no_top_tie':not b['baseline_K1']['top_tie'] and not b['baseline_K2']['top_tie'],'gold':g,'fixed_wrong':wrong,'corners':corners,'metrics':metrics,'candidate_metrics':allc,'metadata_sha256':q['metadata_sha256'],'logits_sha256':q['logits_sha256']})
assert len(items)==1119 and sum(not x['demo'] for x in items)==1094
def summary(rows):
 result={'n':len(rows),'metrics':{k:describe([r['metrics'][k] for r in rows]) for k in items[0]['metrics']}}
 result['all_candidates_equal_question_weight']={k:describe([statistics.mean(v[k] for v in r['candidate_metrics'].values()) for r in rows]) for k in items[0]['metrics']}
 result['prediction']={k:{'correct':sum(r['corners'][k]['prediction']==r['gold'] for r in rows),'top_ties':sum(r['corners'][k]['top_tie'] for r in rows)} for k in NAMES}
 result['transitions_vs_222']={k:{'rescued_by_222':sum(r['corners'][k]['prediction']!=r['gold'] and r['corners']['222']['prediction']==r['gold'] for r in rows),'damaged_by_222':sum(r['corners'][k]['prediction']==r['gold'] and r['corners']['222']['prediction']!=r['gold'] for r in rows)} for k in NAMES}
 return result
with gzip.open(OUT/'train_rows.jsonl.gz','xt',encoding='utf8') as f:
 for r in items:f.write(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n')
report={'status':'TRAIN_EXPLORATORY_ONLY_VAL_TEST_BLINDED','coverage_sha256':sha(cv),'amendment_sha256':sha(ROOT/'research_plans/ROUND_3_TRAIN_FIRST_DISCLOSURE_AMENDMENT.md'),'script_sha256':sha(Path(__file__)),'old_coverage_sha256':sha(OLD/'coverage_train.json'),'binding_sha256':c['binding_sha256'],'rows_sha256':sha(OUT/'train_rows.jsonl.gz'),'all':summary(items),'non_demo':summary([r for r in items if not r['demo']])}
with (OUT/'train_summary.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print('TRAIN_EXPLORATORY_READY',sha(OUT/'train_summary.json'),report['rows_sha256'],flush=True)
