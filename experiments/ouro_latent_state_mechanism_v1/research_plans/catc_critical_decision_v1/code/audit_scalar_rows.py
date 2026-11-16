"""Independent CPU arithmetic, dose, cohort and input-identity audit."""
import pathlib,json,hashlib,math,os,gzip,datetime
R=pathlib.Path(os.environ.get('CATC_CRITICAL_ROOT','/root/ouro-catc-critical-sol-20261116'));rows=[json.loads(l) for l in (R/'QUESTION_LEVEL_RESULTS.jsonl').read_text().splitlines()];f=json.loads((R/'FREEZE.json').read_text());seen=set();hashcohorts={};record_count=0;endpoint_count=0
for q in rows:
 assert q['id'] not in seen;seen.add(q['id']);hashcohorts.setdefault(q['input_ids_sha256'],[]).append((q['id'],q['cohort']));g=q['gold'];assert 0<=g<len(q['native']['scores']['4']['candidate_logits'])
 records=[q['native'],q['selected'],*q['controls'].values(),*q.get('fixed_direction_methods',{}).values()]
 for r in records:
  record_count+=1;expected_sites=set() if not r['norms'] else set(map(str,range(9,16)));assert set(map(str,r['directly_changed_K2_R_sites']))==expected_sites
  assert len(r['norms']) in [0,7]
  for dose in r['norms']:assert dose[3]<=dose[1] and dose[3]>=0
  for d,z in r['scores'].items():
   endpoint_count+=1;v=z['candidate_logits'];assert 3<=len(v)<=5 and all(math.isfinite(x) for x in v);ties=[i for i,x in enumerate(v) if x==max(v)];assert ties==z['ties'] and z['prediction']==ties[0];assert len(z['logits_sha256'])==64;assert all(abs((z['scores'][i]-z['scores'][0])-(v[i]-v[0]))<=2e-5 for i in range(len(v)))
 for kind,r in q['controls'].items():
  assert q['cohort']=='confirmation' and len(r['dose_matches'])==7
  for j,z in enumerate(r['dose_matches']):assert z['target_actual_norm']==q['selected']['norms'][j][3] and abs(z['actual_norm']-r['norms'][j][3])==0 and z['actual_norm']<=z['cap']
for c,ids in f['cohorts'].items():assert set(ids)=={q['id'] for q in rows if q['cohort']==c}
assert sum(q['dataset']=='validation' for q in rows)==299
cross=[v for v in hashcohorts.values() if len({c for _,c in v})>1];repair=json.loads((R/'COHORT_DUPLICATE_REPAIR.json').read_text());excluded=set(repair['excluded_confirmation_ids']);clean=[q for q in rows if q['id'] not in excluded];devhash={q['input_ids_sha256'] for q in clean if q['cohort']=='development'};assert not any(q['input_ids_sha256'] in devhash for q in clean if q['cohort']=='confirmation')
input_count=0
for name in ['TRAIN_INPUTS.jsonl.gz','VALIDATION_INPUTS.jsonl.gz']:
 p=R/name
 if p.exists():
  with gzip.open(p,'rt') as h:
   for line in h:
    q=json.loads(line);digest=hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest();assert digest==q['input_ids_sha256'];assert any(x['id']==q['id'] and x['input_ids_sha256']==digest for x in rows);input_count+=1
out={'status':'PASS','questions':len(rows),'record_visits_including_diagnostic_aliases':record_count,'candidate_endpoint_visits':endpoint_count,'input_sequences_verified':input_count,'exact_prompt_cross_cohort_duplicates':cross,'scope':'all stored candidate predictions/ties/logprob differences, intended sites, actual norm cap and control targets; full-input token hashes when packaged inputs present','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'INDEPENDENT_CPU_AUDIT.json').write_text(json.dumps(out,indent=2));print('INDEPENDENT_SCALAR_AUDIT_PASS',len(rows),endpoint_count,input_count)
