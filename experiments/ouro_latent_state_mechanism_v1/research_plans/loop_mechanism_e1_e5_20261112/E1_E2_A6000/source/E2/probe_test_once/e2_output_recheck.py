import json,collections,math,pathlib,hashlib
p=pathlib.Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E2/probe_test_once'); summary=json.load(open(p/'summary.json'))['results']; R=collections.defaultdict(dict)
for line in open(p/'per_question.jsonl'):
 x=json.loads(line);k=x.pop('model');assert x['id'] not in R[k];R[k][x['id']]=x
assert len(R)==37 and all(len(x)==1172 for x in R.values())
for k,entries in R.items():
 v=list(entries.values());assert all(x.get('status','OK')=='OK' for x in v);assert abs(sum(x['correct'] for x in v)/1172-summary[k]['all']['accuracy'])<1e-14
 for x in v:
  assert abs(sum(x['candidate_probabilities'])-1)<1e-12;assert x['prediction_index']==x['top_ties'][0] and x['gold_index']<len(x['targets']);val=x['candidate_values']; gold=x['gold_index'];assert abs((val[gold]-max(val[j] for j in range(len(val)) if j!=gold))-x['dynamic_gap'])<1e-10
out={}
for key in R:
 if not key.endswith('_evalK1'):continue
 other=key[:-1]+'2';assert set(R[key])==set(R[other]);out[key[:-7]]={}
 for group in ('all','rescue','damage','stable_correct','stable_wrong'):
  for condition in ('all','natural_no_top_tie'):
   pairs=[(a,R[other][ident]) for ident,a in R[key].items() if (group=='all' or a['group']==group) and (condition=='all' or a['natural_no_top_tie'])];res=sum(not a['correct'] and b['correct'] for a,b in pairs);dam=sum(a['correct'] and not b['correct'] for a,b in pairs);out[key[:-7]][group+'__'+condition]={'n':len(pairs),'probe_rescue':res,'probe_damage':dam,'net':res-dam,'stable_correct':sum(a['correct'] and b['correct'] for a,b in pairs),'stable_wrong':sum(not a['correct'] and not b['correct'] for a,b in pairs)}
f=p/'cross_loop_transition_recheck.json';assert not f.exists();f.write_text(json.dumps({'scope':'Postprocessing frozen per-question outputs only; no new test prediction or model selection.','source_per_question_sha256':hashlib.sha256((p/'per_question.jsonl').read_bytes()).hexdigest(),'models':out},sort_keys=True)+'\n');print('PASS 43364 unique rows, normalized candidate probs, argmax/tie, dynamic margin, accuracy summaries; cross_loop_sha256',hashlib.sha256(f.read_bytes()).hexdigest());
for k in ('post_norm_raw_trainK1_real','l18_input_raw_trainK1_real'):print(k,out[k]['all__all'])
