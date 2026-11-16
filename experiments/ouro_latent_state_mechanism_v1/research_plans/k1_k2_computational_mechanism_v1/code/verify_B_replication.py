import pathlib,json,hashlib,datetime
R=pathlib.Path('/root/ouro-computation-sol-20261116');checks=0;manifest=[]
for p in sorted((R/'data/B').glob('*.json')):
 a=json.loads(p.read_text());q=R/'data/B_replicated'/p.name;b=json.loads(q.read_text());assert a['base_pair_id']==b['base_pair_id'] and len(a['units'])==len(b['units'])
 for x,y in zip(a['units'],b['units']):
  assert x['unit_id']==y['unit_id']
  for old,new in zip(x['records'],y['records']):
   for k in ['condition','input_ids_sha256','logits_sha256','candidate_logits','scores','prediction','ties','stage_R_sha256']:assert old[k]==new[k],(x['unit_id'],old['condition'],k)
   checks+=1
 t=q.with_suffix('.pt');d=t.read_bytes();manifest.append({'path':str(t.relative_to(R)),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()})
assert checks==6400
(R/'B_FULL_ARCHIVE_REPLICATION_PASS.json').write_text(json.dumps({'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'units':640,'full_endpoint_records':checks,'all_original_native_and_intervention_outputs_exact':True,'also_exact':'candidate probabilities/scores, predictions, ties, input hashes and every captured K2 R-stage hash','purpose':'Full independent archive rerun using corrected native::/patch:: serializer; preserve originals, retain complete native B states and all intervention tensors. Repairs original missing native serialization without replacing original scientific records.','corrected_raw_archive':manifest},indent=2));print('B_ALL640_FULL_ARCHIVE_REPLICATION_EXACT',checks)
