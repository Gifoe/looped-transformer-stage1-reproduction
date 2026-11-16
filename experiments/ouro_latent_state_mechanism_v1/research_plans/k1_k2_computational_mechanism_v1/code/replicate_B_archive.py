import core,json,sys,hashlib,time,datetime
from core import ROOT,torch,forward,record,cpu,bits,jsave,tsave
rows={q['id']:q for q in map(json.loads,(ROOT/'B_INPUTS.jsonl').read_text().splitlines())};units=list(map(json.loads,(ROOT/'B_UNITS.jsonl').read_text().splitlines()));phase=sys.argv[1];model,lock=core.initialize(worker="B");(ROOT/'data/B_replicated').mkdir(exist_ok=True,parents=True)
def execute(group,technical=False):
 ids=sorted({u[k] for u in group for k in ('receiver','donor','text_edit','same_variable')});native={};result={'units':[]};tensors={}
 for id in ids:
  q=rows[id];z1,z2,n=forward(model,q,mode='native',audit=technical);native[id]=(z1,z2,n)
  tensors['native::'+id]={'B':cpu(n['B']),'K1_logits':cpu(z1),'native_logits':cpu(z2),'stages':{k:cpu(v) for k,v in n['stages'].items()}}
 for u in group:
  qr,qd,qe,qs=[rows[u[k]] for k in ('receiver','donor','text_edit','same_variable')];r=native[qr['id']];d=native[qd['id']];e=native[qe['id']];s=native[qs['id']];lo,hi=qr['source_span'];R=qr['R'];un=qr['unrelated_span']
  assert bits(r[2]['B'][:,lo:hi],s[2]['B'][:,lo:hi]),'causal source changed under future-only mapping edit'
  assert bits(d[2]['B'][:,lo:hi],e[2]['B'][:,lo:hi]),'causal source inconsistent with textual edit'
  records=[record(qr,'K1',r[0],r[2]),record(qr,'native',r[1],r[2]),record(qe,'text_edit_K1',e[0],e[2]),record(qe,'text_edit_K2',e[1],e[2])]
  gen=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(('B-dose-v1|'+u['unit_id']).encode()).digest()[:8],'big')&((1<<63)-1));perm=torch.randperm(2048,generator=gen).cuda();delta=d[2]['B'][:,lo:hi].float()-r[2]['B'][:,lo:hi].float();dose=(r[2]['B'][:,lo:hi].float()+delta[...,perm]).to(torch.bfloat16)
  operations=[('source_boundary',(lo,hi),d[2]['B'][:,lo:hi]),('same_variable',(lo,hi),s[2]['B'][:,lo:hi]),('unrelated',tuple(un),d[2]['B'][:,un[0]:un[1]]),('dose',(lo,hi),dose),('readout_copy',(R,R+1),d[2]['B'][:,R:R+1]),('descendant_edit',(lo,R),e[2]['B'][:,lo:R])]
  patches={}
  for condition,span,donor in operations:
   _,z,n=forward(model,qr,mode='boundary',span=span,donor=donor,audit=technical)
   assert bits(n['B'],r[2]['B'])
   if condition=='same_variable':assert bits(z,r[1]),'same-variable null must be exact'
   if technical:
    before=n['applied'][0];b=cpu(r[2]['B']);mask=torch.ones(R+1,dtype=torch.bool);mask[span[0]:span[1]]=False;assert bits(before[:,mask],b[:,mask])
   records.append(record(qr,condition,z,n));patches[condition]={'logits':cpu(z),'stages':{k:cpu(v) for k,v in n['stages'].items()}}
  dose_stats={'requested_source_row_norm':delta.norm(dim=-1).cpu().tolist(),'realized_unrelated_row_norm':(dose.float()-r[2]['B'][:,lo:hi].float()).norm(dim=-1).cpu().tolist(),'fixed_feature_permutation_sha256':core.digest(perm),'delta_is_zero':bool((delta==0).all())}
  result['units'].append({**u,'causal_future_edit_source_bitwise_equal':True,'records':records,'dose':dose_stats});tensors['patch::'+u['unit_id']]=patches
 result['base_pair_id']=group[0]['pair_id'];tsave(ROOT/'data/B_replicated'/f'{result["base_pair_id"]}.pt',tensors);jsave(ROOT/'data/B_replicated'/f'{result["base_pair_id"]}.json',result)
 return result
if phase=='technical':
 # Fixed first development base-pair, no science selection. All directions+labels.
 group=[u for u in units if u['pair_id']=='CW000'];calls=[];original_sdpa=torch.nn.functional.scaled_dot_product_attention
 def spy(*args,**kw):
  assert kw.get('is_causal') is True and kw.get('attn_mask') is None
  calls.append({'query_shape':list(args[0].shape),'is_causal':kw['is_causal'],'attention_mask':None});return original_sdpa(*args,**kw)
 torch.nn.functional.scaled_dot_product_attention=spy
 try:
  q=rows[group[0]['receiver']];z1,z2,n=forward(model,q,mode='capture',audit=True);_,zr,_=forward(model,q,mode='replay',history=n['history']);assert bits(z2,zr)
  with torch.inference_mode():
   inp=torch.tensor([q['ids']],device='cuda');official=model(inp,use_cache=False,exit_at_step=1).logits[:,q['R']];assert bits(official,z2)
  v=execute(group,True)
 finally:torch.nn.functional.scaled_dot_product_attention=original_sdpa
 jsave(ROOT/'B_SDPA_ACTUAL_CALL_AUDIT.json',{'status':'PASS','calls':calls,'full49152_replay':True,'official_endpoint':True,'future_visibility':'every actual call is causal, no sliding, source precedes all relation mappings'})
 jsave(ROOT/'B_TECHNICAL_PASS.json',{'status':'PASS','base_pair':'CW000','same_variable_full49152_null':True,'future_only_mapping_source_bits_equal':True,'all_other_rows_unchanged':True,'source_causal_precedes_mapping':True,'mask_type':'config24 full-attention, no sliding; actual native mask None with original SDPA is_causal path; source-before-mapping invariant checked under future edits','conditions':json.loads((ROOT/'B_FREEZE.json').read_text())['conditions']});print('B_TECHNICAL_PASS',flush=True)
else:
 assert (ROOT/'B_TECHNICAL_PASS.json').exists();selected=[u for u in units if u['split']==phase];pids=sorted({u['pair_id'] for u in selected})
 for i,pid in enumerate(pids):
  if (ROOT/'data/B_replicated'/f'{pid}.json').exists():continue
  result=execute([u for u in selected if u['pair_id']==pid]);print(json.dumps({'phase':phase,'base_pairs_done':i+1,'total':len(pids),'id':pid,'gpu_seconds':round(sum(r['seconds'] for u in result['units'] for r in u['records']),3)}),flush=True)
 jsave(ROOT/f'B_REPLICATION_{phase}_COMPLETE.json',{'pairs':len(pids),'units':len(selected),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
