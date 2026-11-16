"""Independent scalar and symbolic audit of the released full population."""
import pathlib,json,os,hashlib,math,datetime
R=pathlib.Path(os.environ.get('EDGE_ROOT','/root/ouro-causal-edge-sol-20261117'));qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};rows=[json.loads(l) for l in (R/'ALL_QUESTION_RESULTS.jsonl').read_text().splitlines()];assert len(rows)==384;seen=set();count=0;matched=unmatched=0
for q in qs.values():assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256'] and q['source_span'][1]<=q['M']<q['R']==len(q['ids'])-1
for u in rows:
 assert u['unit_id'] not in seen;seen.add(u['unit_id']);qr,qd,qe=[qs[u[k]] for k in ['receiver','donor','text_edit']];assert qr['mappings'][qd['intermediate']]==u['predicted_cross_cabinet'];assert u['predicted_cross_index']==qr['options'].index(u['predicted_cross_cabinet'])==qe['correct_index'];assert qr['source_span']==qd['source_span'] and len(qr['ids'])==len(qd['ids']);assert len(u['candidates'])==9
 for name,conditions in u['candidates'].items():
  assert set(conditions)=={'transfer','same_variable','unrelated','dose'}
  for c,r in conditions.items():
   for d,z in r['scores'].items():
    count+=1;v=z['candidate_logits'];assert len(v)==4 and all(math.isfinite(x) for x in v);ties=[i for i,x in enumerate(v) if x==max(v)];assert ties==z['ties'] and z['prediction']==ties[0];assert all(abs((z['candidate_logprobs'][i]-z['candidate_logprobs'][0])-(v[i]-v[0]))<2e-5 for i in range(4))
   assert r['scores']['1']['full49152_sha256']==u['native_receiver']['scores']['1']['full49152_sha256'];assert len(r['edits'])==1;e=r['edits'][0];assert e['pass']==2 and e['physical_layer'] in [6,10,14] and e['reconstruction_relative_error']<.01
   if c=='same_variable':assert e['null_identity'] and r['scores']['2']['full49152_sha256']==u['native_receiver']['scores']['2']['full49152_sha256']
   dose=e['dose']
   if dose is not None:assert abs(dose['actual']-e['actual_delta_norm'])<1e-10;assert dose['matched']==(abs(dose['actual']-dose['target'])<=max(1e-9,.01*dose['target']));matched+=dose['matched'];unmatched+=not dose['matched']
assert len({u['pair_id'] for u in rows if u['split']=='development'})==32 and len({u['pair_id'] for u in rows if u['split']=='fresh_confirmation'})==64
oldhash={qs[u['receiver']]['input_ids_sha256'] for u in rows if u['split']=='development'};newhash={qs[u['receiver']]['input_ids_sha256'] for u in rows if u['split']=='fresh_confirmation'};assert not oldhash&newhash
out={'status':'PASS','units':len(rows),'candidate_endpoint_visits':count,'matched_nonzero_dose_cases':matched,'unmatched_nonzero_dose_cases_retained':unmatched,'development_fresh_input_identity_disjoint':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'CPU_SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2));print('EDGE_CPU_SCALAR_PASS',out)
