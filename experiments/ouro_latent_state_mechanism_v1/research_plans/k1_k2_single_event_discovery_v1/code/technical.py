import single_event as O
from single_event import torch
from guard import verify
import json,pathlib,datetime
R=O.ROOT;freeze=verify();qs=list(map(json.loads,(R/'DEV_INPUTS.jsonl').read_text().splitlines()));m=O.initialize('technical');records=[];raw={};witnesses={}
for pid,layer in [('SR000',9),('SR004',12),('SR008',16)]:
 q=next(q for q in qs if q['pair_id']==pid);prior=json.loads((O.PARENT/'native_results'/(pid+'.json')).read_text());anchor=next(x for x in prior['rows'] if x['id']==q['id']);prefix=prior['canonical_native_prefixes'][anchor['prefix_key']]['hashes'];native,nh,pref,_=O.run(m,q,layer,'native',capture=True);assert pref==prefix;assert all(native['scores'][d]['full49152_sha256']==anchor['native']['scores'][d]['full49152_sha256'] for d in ['1','2']);conditions={'native':native};raw[q['id']]={'native':{d:x.cpu() for d,x in nh.items()}}
 for kind in ['self','semantic','random']:
  targets={str(layer):conditions['semantic']['audit']['events'][0]['actual_delta_norms']} if kind=='random' else None;z,h,_,w=O.run(m,q,layer,kind,prefix=prefix,witness=True,targets=targets);conditions[kind]=z;raw[q['id']][kind]={d:x.cpu() for d,x in h.items()};witnesses[q['id']+'_'+kind]={'q':q,'layer':layer,'events':w,'audit':z['audit']};assert O.B.bits(h[1],nh[1]);assert z['audit']['prefix_identity_sites']==194
  if kind=='self':assert all(O.B.bits(h[d],nh[d]) for d in [1,2])
 records.append({'id':q['id'],'layer':layer,'conditions':conditions});print('SINGLE_EVENT_TECHNICAL_CASE',pid,layer,flush=True)
torch.save(raw,O.RAW/'technical_heads.pt');torch.save(witnesses,O.RAW/'technical_witnesses.pt');(R/'TECHNICAL_GPU.json').write_text(json.dumps({'status':'GPU_PASS_PENDING_CPU_ORACLE','freeze_sha256':freeze,'body_forwards':12,'cases':records,'identity_scope':'selfconstructionnull, parentnativefullhashanchor, oneactualK2event/allH/wholemaprecords','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
