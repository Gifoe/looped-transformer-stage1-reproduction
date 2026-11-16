"""Independent CPU validation of every retained full-vocabulary endpoint and symbolic unit."""
import pathlib,json,torch,numpy as np,hashlib,datetime
R=pathlib.Path('/root/ouro-causal-edge-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-causal-edge-sol-20261117/raw');torch.set_num_threads(4);qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};checked=0;maxerr=0.;references=[]
def check(z,score,targets):
 global checked,maxerr
 raw=z.contiguous().view(torch.uint8).numpy().tobytes();assert hashlib.sha256(raw).hexdigest()==score['full49152_sha256'];v=z.reshape(-1).float().numpy().astype(np.float64);assert len(v)==49152 and np.isfinite(v).all();cand=v[targets].tolist();assert cand==score['candidate_logits'];ties=np.flatnonzero(np.asarray(cand)==max(cand)).tolist();assert ties==score['ties'] and score['prediction']==ties[0];a=v.max();lse=a+np.log(np.exp(v-a).sum());err=max(abs(x-(v[t]-lse)) for x,t in zip(score['candidate_logprobs'],targets));assert err<2e-5;maxerr=max(err,maxerr);checked+=1
for p in sorted((R/'results').glob('*.json')):
 q=json.loads(p.read_text());t=torch.load(RAW/(p.stem+'.pt'),map_location='cpu',weights_only=False)
 for id,r in q['native_worlds'].items():
  for d,z in t['native::'+id].items():check(z,r['scores'][str(d)],qs[id]['targets'])
 for u in q['units']:
  qr,qd,qe=[qs[u[k]] for k in ['receiver','donor','text_edit']];symbol=qr['mappings'][qd['intermediate']];assert symbol==u['predicted_cross_cabinet'] and qr['options'].index(symbol)==u['predicted_cross_index']==qe['correct_index'];assert qr['M']==qd['M'] and qr['source_span']==qd['source_span'] and qr['source_span'][1]<=qr['M']<qr['R'];assert len({u['predicted_cross_index'],u['original_index'],u['donor_index']})==3
  for name,c in u['candidates'].items():
   for condition,r in c.items():
    check(t['patch::'+u['unit_id']][name+':'+condition],r['scores']['2'],qr['targets']);assert r['scores']['1']['full49152_sha256']==u['native_receiver']['scores']['1']['full49152_sha256'];e=r['edits'][0];assert e['pass']==2 and e['physical_layer'] in [6,10,14] and e['target']==(qr['M'] if name.startswith('SM') else qr['R']);assert e['reconstruction_relative_error']<.01
    if condition=='same_variable':assert r['scores']['2']['full49152_sha256']==u['native_receiver']['scores']['2']['full49152_sha256'] and e['null_identity']
  if u['split']=='development':
   old=json.loads((pathlib.Path('/root/ouro-computation-sol-20261116/data/B_replicated')/p.name).read_text());v=next(x for x in old['units'] if x['unit_id']==u['unit_id']);rec={x['condition']:x for x in v['records']};assert rec['native']['logits_sha256']==u['native_receiver']['scores']['2']['full49152_sha256'] and rec['K1']['logits_sha256']==u['native_receiver']['scores']['1']['full49152_sha256'];references.append({'unit_id':u['unit_id'],'predicted_cross_index':u['predicted_cross_index'],'native':rec['native'],'source_boundary':rec['source_boundary'],'source_archive':'data/B_replicated/'+p.name,'actual_same_device_native_fullhash_match':True})
(R/'LOCALIZED_BOUNDARY_REFERENCE.json').write_text(json.dumps(references,indent=2));(R/'CPU_FULL_VECTOR_AUDIT.json').write_text(json.dumps({'status':'PASS','full49152_endpoints':checked,'max_candidate_logprob_FP64_reference_error':maxerr,'matched_old_development_units':len(references),'scope':'Every native/counterfactual vector hash, candidate score/tie/prediction/logprob, symbolic cross label, source/target positions, unchangedK1 andsame-variablefullnull. Oldboundaryreference usedonlyaftercurrentnativefullhash parity; freshnotpooledwitholdheldout.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('FULL_VECTOR_CPU_PASS',checked,'OLD_REFERENCE',len(references))
