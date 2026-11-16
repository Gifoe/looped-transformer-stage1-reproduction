import bridge as B
from bridge import torch
import pathlib,json,datetime,time,sys
R=B.ROOT;f=json.loads((R/'FREEZE.json').read_text());selection=json.loads((R/'A_BACKGROUND_SELECTION.json').read_text());bg=selection['g1'];assert bg in [0,1];a=json.loads((R/'confirmation_A_SUMMARY.json').read_text());assert a['contrasts']['native' if bg==1 else'capacity']['pass'];phase=sys.argv[1];qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};units=[u for u in map(json.loads,(R/'UNITS.jsonl').read_text().splitlines()) if u['split']==phase];ids=sorted({u['pair_id'] for u in units});chosen=None if phase=='development' else json.loads((R/'B_SELECTION.json').read_text())['mask'];masks=list(range(1,7)) if phase=='development' else [chosen] if chosen not in [0,7,None] else [];outdir=R/'B_results';outdir.mkdir(exist_ok=True);m=B.initialize();deadline=datetime.datetime.fromisoformat(f['deadline_utc']).timestamp();start=time.monotonic()
for i,pid in enumerate(ids):
 p=outdir/(pid+'.json')
 if p.exists():continue
 group=[u for u in units if u['pair_id']==pid];old=json.loads((R/'results'/(pid+'.json')).read_text());prev={u['id']:u for u in old['units']};natives={};raw={}
 if masks:
  for id in sorted({u[k] for u in group for k in ['reference_W0','input_W1']}):
   r,z,c=B.run(m,qs[id],capture=True);assert all(r['scores'][d]['full49152_sha256']==old['native_worlds'][id]['scores'][d]['full49152_sha256'] for d in ['1','2']);natives[id]=(r,z,c)
 records=[]
 for u in group:
  old=prev[u['id']];base={'0':old['conditions']['C10' if bg else'C00'],'7':old['conditions']['C11' if bg else'C01']};raw[u['id']]={}
  for mask in masks:
   if time.time()>deadline:raise RuntimeError('Hard180minute stop')
   w0=natives[u['reference_W0']];w1=natives[u['input_W1']];r,z,c=B.run(m,qs[u['input_W1']],reference=w0[2],native=w1[2],g1=bg,mask2=mask);base[str(mask)]=r;raw[u['id']][mask]={d:x.cpu() for d,x in z.items()}
  records.append({'id':u['id'],'pair_id':pid,'split':phase,'template':u['template'],'mapping':u['mapping'],'direction':u['direction'],'answer0':u['answer0'],'answer1':u['answer1'],'background_g1':bg,'masks':base})
 torch.save(raw,B.RAW/(pid+'_B.pt'));p.write_text(json.dumps({'pair_id':pid,'phase':phase,'units':records,'new_masks':masks}));print('B_GROUP',phase,i+1,len(ids),round(time.monotonic()-start,1),flush=True)
(R/(phase+'_B_COMPLETE.json')).write_text(json.dumps({'status':'COMPLETE','groups':len(ids),'masks':masks,'bg':bg,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}));print('B_COMPLETE',phase)
