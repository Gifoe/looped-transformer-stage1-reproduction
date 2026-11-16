import bridge as B
from bridge import torch
import pathlib,json,datetime,time,sys,hashlib
R=B.ROOT;f=json.loads((R/'FREEZE.json').read_text());assert json.loads((R/'C_TECHNICAL_PASS.json').read_text())['status']=='PASS';gate=json.loads((R/'B_CONFIRMATION_GATE.json').read_text());assert gate['pass'] and gate['mask']==1;bg=gate['bg'];phase=sys.argv[1];qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};units=[u for u in map(json.loads,(R/'UNITS.jsonl').read_text().splitlines()) if u['split']==phase];pids=sorted({u['pair_id'] for u in units});outdir=R/'C_results';outdir.mkdir(exist_ok=True);m=B.initialize();deadline=datetime.datetime.fromisoformat(f['deadline_utc']).timestamp();start=time.monotonic();conditions={'block':[],'restore_early':list(range(8)),'restore_mid':list(range(8,16)),'restore_late':list(range(16,24)),'reverse_early':list(range(8,24)),'reverse_mid':list(range(8))+list(range(16,24)),'reverse_late':list(range(16)),'random':[]}
for i,pid in enumerate(pids):
 p=outdir/(pid+'.json')
 if p.exists():continue
 group=[u for u in units if u['pair_id']==pid];aprev=json.loads((R/'results'/(pid+'.json')).read_text());bprev={u['id']:u for u in json.loads((R/'B_results'/(pid+'.json')).read_text())['units']};native={};raw={};rows=[]
 for id in sorted({u[k] for u in group for k in ['reference_W0','input_W1']}):
  q=qs[id];r,z,c=B.run(m,q,capture=True,extra_span=tuple(q['regions']['map']));assert all(r['scores'][d]['full49152_sha256']==aprev['native_worlds'][id]['scores'][d]['full49152_sha256'] for d in ['1','2']);native[id]=(r,z,c)
 for u in group:
  if time.time()>deadline:raise RuntimeError('Hard180minute stop')
  q=qs[u['input_W1']];w0=native[u['reference_W0']];w1=native[u['input_W1']];span=tuple(q['regions']['map']);Z,zh,zc=B.run(m,q,reference=w0[2],native=w1[2],g1=bg,mask2=0,capture=True,extra_span=span);T,th,tc=B.run(m,q,reference=w0[2],native=w1[2],g1=bg,mask2=1,capture=True,extra_span=span,path={'span':span,'Z':zc,'observe_only':True});assert all(Z['scores'][d]['full49152_sha256']==bprev[u['id']]['masks']['0']['scores'][d]['full49152_sha256'] and T['scores'][d]['full49152_sha256']==bprev[u['id']]['masks']['1']['scores'][d]['full49152_sha256'] for d in ['1','2']);assert all(all(B.bits(a,b) for a,b in zip(kv,w1[2]['extra_kv'][k])) for k,kv in tc['extra_kv'].items());targets={e['layer']-1:e['counterfactual_T_writeback_norms'] for e in T['cut']['path_edits']};records={'Z':Z,'T':T};raw[u['id']]={'Z':{d:x.cpu() for d,x in zh.items()},'T':{d:x.cpu() for d,x in th.items()}}
  for name,layers in conditions.items():
   if time.time()>deadline:raise RuntimeError('Hard180minute stop')
   path={'span':span,'Z':zc,'T':tc,'restore_layers':layers}
   if name=='random':path.update(random=True,unit_id=u['id'],targets=targets)
   r,z,c=B.run(m,q,reference=w0[2],native=w1[2],g1=bg,mask2=1,path=path);assert B.bits(z[1],th[1])
   if name=='block':assert all(B.bits(z[d],zh[d]) for d in [1,2]) and r['cut']['full_path_block_H_norm_exact']
   records[name]=r;raw[u['id']][name]={d:x.cpu() for d,x in z.items()}
  rows.append({'id':u['id'],'pair_id':pid,'split':phase,'template':u['template'],'mapping':u['mapping'],'direction':u['direction'],'answer0':u['answer0'],'answer1':u['answer1'],'background_g1':bg,'G_span':span,'H_span':[span[1],q['R']+1],'conditions':records})
 torch.save(raw,B.RAW/(pid+'_C.pt'));p.write_text(json.dumps({'pair_id':pid,'phase':phase,'units':rows}));print('C_GROUP',phase,i+1,len(pids),round(time.monotonic()-start,1),flush=True)
(R/(phase+'_C_COMPLETE.json')).write_text(json.dumps({'status':'COMPLETE','groups':len(pids),'units':len(units),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}));print('C_COMPLETE',phase)
