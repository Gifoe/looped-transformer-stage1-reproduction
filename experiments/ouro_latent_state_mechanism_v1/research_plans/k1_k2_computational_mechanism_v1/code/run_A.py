import core,sys,json,time,pathlib,datetime,hashlib
from core import torch,ROOT,forward,record,bits,cpu,jsave,tsave
m=json.loads((ROOT/'A_FREEZE.json').read_text());allq={x['id']:x for x in map(json.loads,(ROOT/'A_INPUTS.jsonl').read_text().splitlines())};phase=sys.argv[1]
ids=m['technical'] if phase=='technical' else m['development64'] if phase=='development' else json.loads((ROOT/'A_EXPANSION_FREEZE.json').read_text())[phase]
model,lock=core.initialize();out=ROOT/'data/A';out.mkdir(exist_ok=True,parents=True)
if phase=='technical':
 tests=[]
 for id in ids:
  q=allq[id];z1,z2,n=forward(model,q,mode='capture',audit=True)
  z1b,z2b,n2=forward(model,q,mode='native');assert bits(z1,z1b) and bits(z2,z2b)
  _,zr,rr=forward(model,q,mode='replay',history=n['history'],audit=True);assert bits(z2,zr),'native replay failed'
  with torch.inference_mode():
   ids_t=torch.tensor([q['ids']],device='cuda')
   official1=model(ids_t,use_cache=False,exit_at_step=0).logits[:,q['R']]
   official2=model(ids_t,use_cache=False,exit_at_step=1).logits[:,q['R']]
  assert bits(z1,official1) and bits(z2,official2),'official full shape endpoints fail'
  witness={'native_history':{i:cpu(x) for i,x in n['history'].items()},'B':cpu(n['B']),'K1_logits':cpu(z1),'K2_logits':cpu(z2),'replay_logits':cpu(zr),'mask_args':n.get('mask_args'),'replay_applied':rr['applied']}
  # Controls exercised on every technical shape before scientific cohort.
  control=[]
  for condition in m['conditions']:
   _,z,r=forward(model,q,mode=condition,audit=False)
   control.append(record(q,condition,z,r))
  tsave(ROOT/'technical'/f'{id}.pt',witness)
  tests.append({'id':id,'full49152_repeat':True,'official_K1_K2':True,'full49152_replay_bitwise':True,'native_seconds':n['elapsed_seconds'],'treatment_seconds':[x['seconds'] for x in control],'mask_args':n.get('mask_args'),'controls':control});print(json.dumps({'technical_id':id,'native_seconds':n['elapsed_seconds'],'control_seconds':[round(x['seconds'],3) for x in control]}),flush=True)
 jsave(ROOT/'TECHNICAL_PASS.json',{'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests':tests,'full_shaped_replay':'non-R native layer inputs are restored; full-shaped original attention/MLP run; this is identity gate, not mechanism evidence'})
else:
 assert (ROOT/'TECHNICAL_PASS.json').exists()
 for idx,id in enumerate(ids):
  q=allq[id];path=out/f'{q["split"]}__{id}.json'
  if path.exists():continue
  z1,z2,n=forward(model,q,mode='native');records=[record(q,'K1',z1,n),record(q,'native',z2,n)]
  tensors={'B':cpu(n['B']),'K1_logits':cpu(z1),'native_logits':cpu(z2),'native_stages':{k:cpu(v) for k,v in n['stages'].items()}}
  b_hash=core.digest(n['B'])
  for condition in m['conditions']:
   _,z,r=forward(model,q,mode=condition);assert core.digest(r['B'])==b_hash
   records.append(record(q,condition,z,r));tensors[condition+'_logits']=cpu(z);tensors[condition+'_stages']={k:cpu(v) for k,v in r['stages'].items()}
  tsave(out/f'{q["split"]}__{id}.pt',tensors);jsave(path,{'id':id,'split':q['split'],'phase':phase,'records':records});print(json.dumps({'phase':phase,'done':idx+1,'total':len(ids),'id':id,'gpu_seconds':round(sum(r['seconds'] for r in records[1:]),3)}),flush=True)
 jsave(ROOT/f'A_{phase}_COMPLETE.json',{'count':len(ids),'ids':ids,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
