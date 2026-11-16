import ATTENTION_EDGE_IMPLEMENTATION as E
from ATTENTION_EDGE_IMPLEMENTATION import torch
import pathlib,json,time,datetime,sys,hashlib
R=E.ROOT;f=json.loads((R/'FREEZE.json').read_text());rows={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};units=[json.loads(l) for l in (R/'UNITS.jsonl').read_text().splitlines()];deadline=datetime.datetime.fromisoformat(f['deadline_utc']).timestamp();m=E.initialize();phase=sys.argv[1];technical=phase=='technical';allpairs=sorted({u['pair_id'] for u in units if u['split']==('development' if technical else phase)});allpairs=allpairs[:1] if technical else allpairs
if not technical:assert json.loads((R/'TECHNICAL_PASS.json').read_text())['status']=='PASS'
start=time.monotonic()
for number,pid in enumerate(allpairs):
 path=R/'results'/f'{pid}.json'
 if path.exists() and not technical:continue
 group=[u for u in units if u['pair_id']==pid];ids=sorted({u[k] for u in group for k in ['receiver','donor','text_edit','same_variable','unrelated_source']});native={};cache={};tensors={};receipts=[];native_records={}
 for id in ids:
  if time.time()>deadline:raise RuntimeError('Hard180minbudgetstop')
  q=rows[id];r,z,c,w=E.forward(m,q,capture=True);native[id]=(r,z);cache[id]=c;native_records[id]=r;tensors['native::'+id]={d:x.cpu() for d,x in z.items()}
  if technical:
   inp=torch.tensor([q['ids']],device='cuda')
   with torch.inference_mode():_,states,_=m.model(inp,use_cache=False,past_key_values=None);official={d+1:m.lm_head(states[d])[:,q['R']] for d in range(2)}
   assert all(E.bits(z[d],official[d]) for d in [1,2]);receipts.append({'id':id,'official_all49152_bothpasses_bitwise':True})
 # Causal future invariance at every tested pass/layer, allsame-variable mappingedits.
 causal=[]
 for u in group:
  for a,b in [('receiver','same_variable'),('donor','text_edit')]:
   for ut in [0,1]:
    for layer in E.LAYERS:
     ka,va=cache[u[a]][(ut,layer,'S')];kb,vb=cache[u[b]][(ut,layer,'S')];assert E.bits(ka,kb) and E.bits(va,vb);causal.append({'unit':u['unit_id'],'worlds':[u[a],u[b]],'pass':ut+1,'layer':layer+1,'K_sha256':E.digest(ka),'V_sha256':E.digest(va)})
 # Option-order-only changes cannot influence earlier source K/V.
 for world in ['original','donor','original_edit','donor_edit']:
  a=f'{pid}_L0_{world}';b=f'{pid}_L1_{world}'
  for ut in [0,1]:
   for layer in E.LAYERS:
    assert all(E.bits(x,y) for x,y in zip(cache[a][(ut,layer,'S')],cache[b][(ut,layer,'S')]))
 result=[];all_witness={}
 for u in group:
  qr=rows[u['receiver']];records={};case_tensors={}
  for path_name,definition in f['paths'].items():
   layer=definition['layer']-1
   for kind in f['transfers']:
    name=path_name+'_'+kind;conditions={}
    for condition,key in [('transfer','donor'),('same_variable','same_variable'),('unrelated','unrelated_source'),('dose','donor')]:
     if time.time()>deadline:raise RuntimeError('Hard180minbudgetstop')
     op={'pass':1,'layer':layer,'target':definition['target'],'type':kind,'source':cache[u[key]][(1,layer,'S')],'shuffled':condition=='dose','unit_id':u['unit_id']};r,z,c,w=E.forward(m,qr,[op],witness=technical and u==group[0] and condition=='transfer');assert E.bits(z[1],native[qr['id']][1][1]);conditions[condition]=r;case_tensors[name+':'+condition]=z[2].cpu()
     if condition=='same_variable':assert E.bits(z[2],native[qr['id']][1][2]) and r['edits'][0]['null_identity']
     if w:all_witness[name]=w
    records[name]=conditions
  result.append({**u,'native_receiver':native[u['receiver']][0],'native_donor':native[u['donor']][0],'textual_counterfactual':native[u['text_edit']][0],'candidates':records});tensors['patch::'+u['unit_id']]=case_tensors
 out={'pair_id':pid,'phase':'technical' if technical else phase,'units':result,'native_worlds':native_records,'causal_source_future_invariance':causal,'option_order_future_invariance':True};torch.save(tensors,E.RAW/f'{pid}.pt');path.write_text(json.dumps(out))
 if technical:
  torch.save(all_witness,E.RAW/'TECHNICAL_ATTENTION_WITNESSES.pt');torch.save({id:{str(k):tuple(x.cpu() for x in v) for k,v in c.items()} for id,c in cache.items()},E.RAW/'TECHNICAL_SOURCE_KV.pt');runtime=sum(r['seconds'] for r in native_records.values())+sum(r['seconds'] for u in result for c in u['candidates'].values() for r in c.values());gate={'status':'PASS','base_pair':pid,'official_zero_identity':receipts,'source_future_mapping_and_options_bitwise':True,'same_variable_all_condition_null':True,'all_other_attention_rows_byte_equal_asserted':True,'counterfactual_K1_endpoint_unchanged':True,'witness_path':str(E.RAW/'TECHNICAL_ATTENTION_WITNESSES.pt'),'formal_pair_estimated_seconds':runtime,'full96pair_estimated_seconds':runtime*96,'actual_technical_wall_seconds':time.monotonic()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'TECHNICAL_PASS.json').write_text(json.dumps(gate,indent=2));path.rename(R/'TECHNICAL_RESULTS.json');print('EDGE_TECHNICAL_PASS',runtime*96,flush=True)
 else:print('A_PAIR',phase,number+1,len(allpairs),'wall',round(time.monotonic()-start,1),flush=True)
if not technical:(R/(phase+'_COMPLETE.json')).write_text(json.dumps({'status':'COMPLETE','base_pairs':len(allpairs),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}));print('A_PHASE_COMPLETE',phase)
