"""Reciprocal-world design identities; no model run or change to primary endpoints."""
import pathlib,json,os,numpy as np,datetime
R=pathlib.Path(os.environ.get('BRIDGE_ROOT','/root/ouro-temporal-bridge-sol-20261117'));rows=[u for p in sorted((R/'results').glob('TB*.json')) for u in json.loads(p.read_text())['units']] if not (R/'ALL_QUESTION_RESULTS.jsonl').exists() else [json.loads(l) for l in (R/'ALL_QUESTION_RESULTS.jsonl').read_text().splitlines()];lookup={u['id']:u for u in rows};tests=[];by={}
def m(u,c):
 x=u['conditions'][c]['scores']['2']['candidate_logits'];return x[u['answer1']]-x[u['answer0']]
for u in rows:
 if u['direction']!=0:continue
 v=lookup[u['id'][:-1]+'1'];assert u['answer0']==v['answer1'] and u['answer1']==v['answer0'];pairs=[('C00','C11'),('C11','C00'),('C01','C10'),('C10','C01')]
 for a,b in pairs:
  for d in ['1','2']:assert u['conditions'][a]['scores'][d]['full49152_sha256']==v['conditions'][b]['scores'][d]['full49152_sha256']
 a_capacity=m(u,'C01')-m(u,'C00');b_native=m(v,'C11')-m(v,'C10');assert a_capacity==b_native;ia=m(u,'C11')-m(u,'C10')-m(u,'C01')+m(u,'C00');ib=m(v,'C11')-m(v,'C10')-m(v,'C01')+m(v,'C00');assert ia==-ib;tests.append({'unit_forward':u['id'],'unit_reverse':v['id'],'phase':u['split'],'interaction_forward':ia,'interaction_reverse':ib});by.setdefault(u['split'],[]).append(ia)
out={'status':'PASS','reciprocal_pairs':len(tests),'phases':{p:{'pairs':len(v),'forward_mean_interaction':float(np.mean(v)),'reverse_mean_interaction':float(-np.mean(v)),'mean_abs_interaction':float(np.mean(np.abs(v))),'max_abs_interaction':float(np.max(np.abs(v)))} for p,v in by.items()},'tests':tests,'interpretation':'Channelcuts pluscausalSprefix makeC01forward byte-equalC10reverse atbothRendpoints. Signedtargetmargin reverses too. Thus pooledA-capacity=A-native andpooledsignedinteraction0 byconstruction, not independentclaims or additivityevidence. Perorientation outcomes retain conditionalmeaning. No endpoints/gates/weights changed.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'TEMPORAL_SYMMETRY_AUDIT.json').write_text(json.dumps(out,indent=2));print('RECIPROCAL_DESIGN_IDENTITY_PASS',out['phases'])
