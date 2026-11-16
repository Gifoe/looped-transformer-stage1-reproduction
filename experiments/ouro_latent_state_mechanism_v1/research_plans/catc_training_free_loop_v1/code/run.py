import CATC_IMPLEMENTATION as C
from CATC_IMPLEMENTATION import torch,ROOT,save,sha
import json,sys,time,fcntl,datetime
phase=sys.argv[1];worker=int(sys.argv[2]) if len(sys.argv)>2 else 0;lock=open(ROOT/f'worker{worker}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);f=json.loads((ROOT/'FREEZE.json').read_text());allq={q['id']:q for q in map(json.loads,(ROOT/'INPUTS.jsonl').read_text().splitlines())};model=C.initialize();start=time.monotonic()
if phase=='technical':
 tests=[]
 for id in f['technical_ids']:
  q=allq[id];base,bz,_=C.forward(model,q);zero,zz,w=C.forward(model,q,audit=True)
  assert all(torch.equal(bz[k].view(torch.uint8),zz[k].view(torch.uint8)) for k in range(1,5))
  with torch.inference_mode():
   ids=torch.tensor([q['ids']],device='cuda');official=model(ids,use_cache=False,exit_at_step=3).logits[:,q['R']].cpu();assert torch.equal(official.view(torch.uint8),bz[4].view(torch.uint8))
  torch.save({'full_logits':bz,'audit_witness':w['witness']},ROOT/'technical'/f'{id}.pt')
  times=[]
  for m in f['methods'][1:]:
   rr,rz,rec=C.forward(model,q,**{k:m[k] for k in ['av','ap','kind']});times.append(rr['seconds'])
  tests.append({'id':id,'native_seconds':base['seconds'],'treatment_seconds':times,'zero_full49152_all4_bitwise':True,'officialK4_full49152_bitwise':True,'audits':w['audit'],'peak_allocated_bytes':base['peak_allocated_bytes']});print('TECH',id,base['seconds'],sum(times),flush=True)
 save('ATTENTION_DECOMPOSITION_AUDIT.json',{'status':'PASS','tests':tests,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
else:
 assert (ROOT/'ATTENTION_DECOMPOSITION_AUDIT.json').exists()
 for techid in f['technical_ids']:
  rq,rz,_=C.forward(model,allq[techid]);ref=torch.load(ROOT/'technical'/f'{techid}.pt',weights_only=False)['full_logits'];assert all(torch.equal(rz[k].view(torch.uint8),ref[k].view(torch.uint8)) for k in ref)
 save(f'WORKER{worker}_NATIVE_IDENTITY_PASS.json',{'status':'PASS','all3technical4depthfull49152':True})
 nworkers=int(sys.argv[3]) if len(sys.argv)>3 else 1;ids=f['ids'][worker::nworkers]
 for i,id in enumerate(ids):
  p=ROOT/'results'/f'{id}.json'
  if p.exists():continue
  q=allq[id];records=[];tensors={}
  for method in f['methods']:
   rr,rz,_=C.forward(model,q,**{k:method[k] for k in ['av','ap','kind']});records.append({**method,**rr});tensors[method['name']]=rz
  tmp=p.with_suffix('.tmp');torch.save(tensors,p.with_suffix('.pt'));tmp.write_text(json.dumps({'id':id,'split':'train','records':records}));tmp.replace(p);print(json.dumps({'worker':worker,'done':i+1,'total':len(ids),'id':id,'forward_seconds':sum(x['seconds'] for x in records)}),flush=True)
 save(f'WORKER{worker}_COMPLETE.json',{'n':len(ids),'elapsed':time.monotonic()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
