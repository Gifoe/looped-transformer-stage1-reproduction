import bridge as B
from bridge import torch
import pathlib,json,datetime,time,sys
R=B.ROOT;f=json.loads((R/'FREEZE.json').read_text());assert json.loads((R/'CAUSAL_CUT_NUMERICAL_AUDIT.json').read_text())['status']=='PASS';phase=sys.argv[1];qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};units=[u for u in map(json.loads,(R/'UNITS.jsonl').read_text().splitlines()) if u['split']==phase];pids=sorted({u['pair_id'] for u in units});deadline=datetime.datetime.fromisoformat(f['deadline_utc']).timestamp();m=B.initialize();start=time.monotonic()
for i,pid in enumerate(pids):
 out=R/'results'/f'{pid}.json'
 if out.exists():continue
 group=[u for u in units if u['pair_id']==pid];ids=sorted({u[k] for u in group for k in ['reference_W0','input_W1']});native={};tensors={};provenance={};canonical={};sources={}
 for id in ids:
  if time.time()>deadline:raise RuntimeError('Hard180minute stop; preservecompletedgroups')
  r,z,c=B.run(m,qs[id],capture=True);native[id]=(r,z,c);tensors['native::'+id]={d:v.cpu() for d,v in z.items()};variable=qs[id]['intermediate']
  if variable not in canonical:
   canonical[variable]=id;sources[id]={str(k):{'K':B.digest(kv[0]),'V':B.digest(kv[1])} for k,kv in c['kv'].items()}
  else:
   ref=native[canonical[variable]][2];assert all(all(B.bits(a,b) for a,b in zip(kv,ref['kv'][k])) for k,kv in c['kv'].items())
  provenance[id]={'actual_reference_world':id,'identical_prefix_canonical_world':canonical[variable],'fullinput_sha256':qs[id]['input_ids_sha256'],'b':qs[id]['b'],'native_fullendpoints':r['scores']}
 results=[]
 for u in group:
  q=qs[u['input_W1']];w0=native[u['reference_W0']];w1=native[u['input_W1']];records={'C11':w1[0]};raw={}
  for name,g1,mask2 in [('C00',0,0),('C01',0,7),('C10',1,0)]:
   if time.time()>deadline:raise RuntimeError('Hard180minute stop')
   r,z,c=B.run(m,q,reference=w0[2],native=w1[2],g1=g1,mask2=mask2);assert B.bits(z[1],w0[1][1] if not g1 else w1[1][1])
   if name=='C00':assert all(B.bits(z[d],w0[1][d]) for d in [1,2])
   records[name]=r;raw[name]={d:v.cpu() for d,v in z.items()}
  tensors['unit::'+u['id']]=raw;results.append({**u,'native_W0':w0[0],'native_W1':w1[0],'conditions':records})
 torch.save(tensors,B.RAW/(pid+'_A.pt'));out.write_text(json.dumps({'pair_id':pid,'phase':phase,'units':results,'native_worlds':{id:v[0] for id,v in native.items()},'reference_provenance':provenance,'canonical_source_KV_hashes':sources}));print('A_GROUP',phase,i+1,len(pids),'wall',round(time.monotonic()-start,1),flush=True)
(R/(phase+'_A_COMPLETE.json')).write_text(json.dumps({'status':'COMPLETE','groups':len(pids),'units':len(units),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}));print('A_COMPLETE',phase)
