import model
import region_operator as O
from guard import verify,sha,exact
import json,sys,time,torch,datetime
R=model.R;P=model.P;D=R.parent/'ouro-binding-interface-sol-20261117';fs=verify();assert json.loads((R/'TECHNICAL_GATE.json').read_text())['status']=='PASS';rank=int(sys.argv[1]);assert rank in range(3);qs=[q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines()) if q['split']=='development' and int(q['pair_id'][2:])%3==rank];m=model.initialize('dev'+str(rank));start=time.monotonic()
for pid in sorted({q['pair_id'] for q in qs}):
 prior=json.loads((D/'results'/(pid+'_L11.json')).read_text());full={u['id']:u for u in prior['rows']};native=json.loads((P/'native_results'/(pid+'.json')).read_text());nd={u['id']:u for u in native['rows']};group=[q for q in qs if q['pair_id']==pid]
 for mask in range(1,7):
  name=pid+f'_M{mask}';path=R/'results'/(name+'.json');rawpath=O.RAW/(name+'.pt')
  if path.exists():
   old=json.loads(path.read_text());assert old['freeze_sha256']==fs and sha(rawpath)==old['raw_sha256'];exact(old['rows'],[q['id'] for q in group]);continue
  rows=[];raw={};label={}
  for q in group:
   a=nd[q['id']];pref=native['canonical_native_prefixes'][a['prefix_key']]['hashes'];sem,sh,_,_=O.run(m,q,kind='semantic',mask=mask,native_prefix=pref);target={'11':sem['audit']['events'][0]['actual_delta_norms']};ran,rh,_,_=O.run(m,q,kind='random',mask=mask,native_prefix=pref,target_doses=target);assert all(rec['scores']['1']==a['native']['scores']['1'] for rec in [sem,ran]);nq=q['Q_span'][1]-q['H_start'];sel=[t for bit,span in zip([1,2,4],[q['Q_span'],q['O_span'],[q['R'],q['R']+1]]) if mask&bit for t in range(*span)];expected=[v if t in sel else 0. for t,v in zip(range(q['H_start'],q['R']+1),full[q['id']]['semantic']['audit']['events'][0]['actual_delta_norms'])];assert sem['audit']['events'][0]['actual_delta_norms']==expected
   key=tuple(q[k] for k in ['firsthop_factor','endpoint_factor','selector','record_order']);hashes=sem['audit']['Q_prefix_hashes']
   if q['label_factor']:assert hashes==label[key],('labelprefix',q['id'])
   else:label[key]=hashes
   meta={k:q[k] for k in ['id','pair_id','template','firsthop_factor','endpoint_factor','selector','label_factor','record_order','correct_index','opposite_index','decoy_indices','Q_span','O_span','H_start','R']};rows.append({**meta,'mask':mask,'native':a['native'],'semantic':sem,'random':ran,'full_semantic':full[q['id']]['semantic'],'full_random':full[q['id']]['random']});raw[q['id']]={'semantic':{d:x.cpu() for d,x in sh.items()},'random':{d:x.cpu() for d,x in rh.items()}}
  torch.save(raw,rawpath);path.write_text(json.dumps({'pair_id':pid,'mask':mask,'freeze_sha256':fs,'raw_sha256':sha(rawpath),'rows':rows}));print('REGION_DEV',rank,name,round(time.monotonic()-start,1),flush=True)
(R/f'DEV_SHARD_{rank}_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','rank':rank,'freeze_sha256':fs,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
