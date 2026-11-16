from freeze_guard import verify,exact_ids,sha
FREEZE_SHA=verify()
import record_operator_v2r1 as O
from record_operator_v2r1 import torch
import pathlib,json,time,datetime,sys
R=O.ROOT;phase=sys.argv[1];cap=json.loads((R/(phase+'_NATIVE_CAPABILITY.json')).read_text());assert cap['pass'] and cap['freeze_sha256']==FREEZE_SHA;qs=[q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines()) if q['split']==phase];outdir=R/'results';outdir.mkdir(exist_ok=True);m=O.initialize();start=time.monotonic()
for j,pid in enumerate(sorted({q['pair_id'] for q in qs})):
 path=outdir/(pid+'.json')
 if path.exists():
  prior=json.loads(path.read_text());assert prior['freeze_sha256']==FREEZE_SHA;exact_ids(prior['rows'],[q['id'] for q in qs if q['pair_id']==pid]);assert sha(O.RAW/(pid+('_native.pt' if 'native_results' in str(path) else '_science.pt')))==prior['raw_sha256'];continue
 native=json.loads((R/'native_results'/(pid+'.json')).read_text());assert native['freeze_sha256']==FREEZE_SHA;exact_ids(native['rows'],[q['id'] for q in qs if q['pair_id']==pid]);nd={u['id']:u for u in native['rows']};rows=[];raw={}
 for q in [q for q in qs if q['pair_id']==pid]:
  anchors=nd[q['id']];prefix=native['canonical_native_prefixes'][anchors['prefix_key']]['hashes'];conditions={'native':anchors['native']};raw[q['id']]={}
  for arm in [1,2]:
   sem,z,_,_=O.run(m,q,kind='semantic',arm=arm,native_prefix=prefix);conditions['semanticK'+str(arm)]=sem;raw[q['id']]['semanticK'+str(arm)]={d:x.cpu() for d,x in z.items()};targets={str(e['layer']):e['actual_delta_norms'] for e in sem['audit']['events']};random,rh,_,_=O.run(m,q,kind='random',arm=arm,native_prefix=prefix,target_doses=targets);conditions['randomK'+str(arm)]=random;raw[q['id']]['randomK'+str(arm)]={d:x.cpu() for d,x in rh.items()}
   if arm==2:assert all(x['scores']['1']['full49152_sha256']==anchors['native']['scores']['1']['full49152_sha256'] for x in [sem,random])
  rows.append({'id':q['id'],'pair_id':pid,'split':phase,'template':q['template'],'firsthop_factor':q['firsthop_factor'],'endpoint_factor':q['endpoint_factor'],'selector':q['selector'],'label_factor':q['label_factor'],'record_order':q['record_order'],'correct_index':q['correct_index'],'opposite_index':q['opposite_index'],'decoy_indices':q['decoy_indices'],'conditions':conditions})
 torch.save(raw,O.RAW/(pid+'_science.pt'));path.write_text(json.dumps({'pair_id':pid,'split':phase,'rows':rows,'freeze_sha256':FREEZE_SHA,'raw_sha256':sha(O.RAW/(pid+'_science.pt'))}));print('SCIENCE_GROUP',phase,j+1,len({q['pair_id'] for q in qs}),round(time.monotonic()-start,1),flush=True)
(R/(phase+'_SCIENCE_COMPLETE.json')).write_text(json.dumps({'freeze_sha256':FREEZE_SHA,'status':'COMPLETE','worlds':len(qs),'groups':len({q['pair_id'] for q in qs}),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
