"""Independent CPU arithmetic, input grouping, strict effects and actual doses."""
import pathlib,json,hashlib,os,math,numpy as np,datetime
R=pathlib.Path(os.environ.get('CCRU_ROOT','/root/ouro-context-update-sol-20261117'));f=json.loads((R/'FREEZE.json').read_text());rows=[json.loads(l) for l in (R/'ALL_QUESTION_RESULTS.jsonl').read_text().splitlines()];inputs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};counts={};endpoint=0;matches=0;unmatched=0
for q in rows:
 x=inputs[q['id']];assert q['input_ids_sha256']==x['input_ids_sha256']==hashlib.sha256(json.dumps(x['ids']).encode()).hexdigest();counts.setdefault(q['cohort'],set()).add(x['input_ids_sha256']);native=q['conditions']['N'];g=q['gold']
 for name,r in q['conditions'].items():
  assert r['scores']['1']['full49152_sha256']==native['scores']['1']['full49152_sha256'];a=r['intervention'];assert a['patch_calls']==int(name!='N')
  if name!='N':
   expected={'Q':(x['prefix_len'],x['R']),'Random':(x['prefix_len'],x['R']),'R':(x['R'],x['R']+1),'P':(0,x['prefix_len'])}[a['region']];assert a['span']==list(expected) and len(a['norms'])==a['tokens']==expected[1]-expected[0]
   for z in a['norms']:assert 0<=z[3]<=z[1] and abs(z[1]-.1*z[4])<1e-12
   assert abs(math.sqrt(sum(z[3]**2 for z in a['norms']))-a['total_displacement_L2'])<=1e-10
   dose=a.get('dose_matching')
   if dose:
    for target,actual,matched in zip(dose['target'],dose['actual'],dose['matched']):assert matched==(abs(actual-target)<=max(1e-9,.01*target));matches+=matched;unmatched+=not matched
  for d,z in r['scores'].items():
   endpoint+=1;v=z['candidate_logits'];assert 3<=len(v)<=5 and 0<=g<len(v);ties=[i for i,x in enumerate(v) if x==max(v)];assert z['ties']==ties and z['prediction']==ties[0];assert all(abs((z['candidate_logprobs'][i]-z['candidate_logprobs'][0])-(v[i]-v[0]))<2e-5 for i in range(len(v)))
assert not counts['development']&counts['confirmation'];summary=json.loads((R/'SUMMARY.json').read_text())
for c in ['development','confirmation']:
 qs=[q for q in rows if q['cohort']==c]
 for n,depths in summary[c]['methods'].items():
  for d,z in depths.items():
   rescue=damage=correct=0
   for q in qs:
    a=q['conditions'][n]['scores'][d];b=q['conditions']['N']['scores'][d];ac=a['prediction']==q['gold'];bc=b['prediction']==q['gold'];correct+=ac
    if len(a['ties'])==len(b['ties'])==1:rescue+=ac and not bc;damage+=bc and not ac
   assert (correct,rescue,damage)==(z['correct'],z['strict_rescue'],z['strict_damage'])
out={'status':'PASS','questions':len(rows),'endpoints':endpoint,'matched_random_token_sites':matches,'unmatched_random_token_sites_retained':unmatched,'exact_input_groups_disjoint':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'INDEPENDENT_CPU_AUDIT.json').write_text(json.dumps(out,indent=2));print('CPU_AUDIT_PASS',out)
