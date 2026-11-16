import single_event as O
from single_event import torch
from guard import verify,sha,exact_ids
import pathlib,json,sys,time,datetime
R=O.ROOT;fsha=verify();assert json.loads((R/'TECHNICAL_GATE.json').read_text())['status']=='PASS';rank=int(sys.argv[1]);workers=int(sys.argv[2]);assert workers==3 and rank in range(workers);allqs=list(map(json.loads,(R/'DEV_INPUTS.jsonl').read_text().splitlines()));qs=[q for q in allqs if int(q['pair_id'][2:])%workers==rank];m=O.initialize(rank);outdir=R/'results';outdir.mkdir(exist_ok=True);start=time.monotonic();done=0
for pid in sorted({q['pair_id'] for q in qs}):
 native=json.loads((O.PARENT/'native_results'/(pid+'.json')).read_text());nd={u['id']:u for u in native['rows']};group=[q for q in qs if q['pair_id']==pid]
 for layer in range(9,17):
  name=pid+f'_L{layer:02d}';path=outdir/(name+'.json');rawpath=O.RAW/(name+'.pt')
  if path.exists():
   old=json.loads(path.read_text());assert old['freeze_sha256']==fsha and old['layer']==layer;exact_ids(old['rows'],[q['id'] for q in group]);assert sha(rawpath)==old['raw_sha256'];continue
  rows=[];raw={}
  for q in group:
   anchor=nd[q['id']];prefix=native['canonical_native_prefixes'][anchor['prefix_key']]['hashes'];sem,sh,_,_=O.run(m,q,layer,'semantic',prefix=prefix);targets={str(layer):sem['audit']['events'][0]['actual_delta_norms']};random,rh,_,_=O.run(m,q,layer,'random',prefix=prefix,targets=targets)
   assert all(rec['scores']['1']['full49152_sha256']==anchor['native']['scores']['1']['full49152_sha256'] for rec in [sem,random]);rows.append({'id':q['id'],'pair_id':pid,'layer':layer,'template':q['template'],'firsthop_factor':q['firsthop_factor'],'endpoint_factor':q['endpoint_factor'],'selector':q['selector'],'label_factor':q['label_factor'],'record_order':q['record_order'],'correct_index':q['correct_index'],'opposite_index':q['opposite_index'],'decoy_indices':q['decoy_indices'],'native':anchor['native'],'semantic':sem,'random':random});raw[q['id']]={'semantic':{d:x.cpu() for d,x in sh.items()},'random':{d:x.cpu() for d,x in rh.items()}}
  torch.save(raw,rawpath);path.write_text(json.dumps({'pair_id':pid,'layer':layer,'freeze_sha256':fsha,'raw_sha256':sha(rawpath),'rows':rows}));done+=1;print('DISCOVERY_CANDIDATE',rank,name,done,round(time.monotonic()-start,1),flush=True)
(R/f'SHARD_{rank}_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','rank':rank,'groups':len({q['pair_id'] for q in qs}),'freeze_sha256':fsha,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
