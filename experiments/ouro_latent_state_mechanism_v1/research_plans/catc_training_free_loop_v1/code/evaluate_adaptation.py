import CATC_ADAPTATION as C
from CATC_ADAPTATION import ROOT,torch
from E6A_METHODS import infer,fixed_specs
import json,pathlib,time,datetime,numpy as np
R=ROOT.parent;deadline=datetime.datetime.fromisoformat(json.loads((ROOT/'FREEZE.json').read_text())['budget_deadline_utc']).timestamp();m=C.initialize();schemes=['catc14','unstructured14','value7','routing7'];selected={n:json.loads((ROOT/n/'COMPLETE.json').read_text()) for n in schemes};best_name=min(['catc14','value7','routing7'],key=lambda n:(selected[n]['validation_CE'],selected[n]['parameters'],selected[n]['best_step'],n));C.save('VALIDATION_MODEL_SELECTION.json',{'selected_name':best_name,'selected':selected[best_name],'scope':'SecondaryValidation-calibratedstructuredfamily, frozenbeforeSciQ/FASoutcomes; primarycatc14retained. Noadditionaloptimizerorcoefficientsearch.'});(ROOT/'SELECTED_COEFFICIENTS.json').write_text(json.dumps(selected,indent=2))
def outputs(r0):
 out={n:infer([r0['scores'][str(k)]['candidate_logits'] for k in range(1,5)],v) for n,v in fixed_specs().items()};x1=np.array(r0['scores']['1']['candidate_logits'],dtype=np.float32);x4=np.array(r0['scores']['4']['candidate_logits'],dtype=np.float32);x=x4+.5*(x4-x1);out['LoopCD_fixed_0.5']={'prediction':int(x.argmax()),'scores':x.tolist(),'scores_units':'F32guidedcandidate_logits'};return out
def call(q,a,kind='structured'):
 if time.time()>deadline:raise RuntimeError('Originalfivehourpilotbudget exhausted, preservecompletequestions')
 r,z,_=C.forward(m,q,kind=kind,coefficients=a);return r
# Validation rows were already evaluated for early stopping; reuse without pretending blindness.
native={x['id']:x for x in json.loads((ROOT/'validation_native.json').read_text())};reuse={}
for n,v in selected.items():
 reuse[n]=native if v['best_step']==0 else {x['id']:x for x in json.loads((ROOT/n/f'validation_step{v["best_step"]}.json').read_text())['rows']}
val=list(map(json.loads,(ROOT/'VALIDATION_INPUTS.jsonl').read_text().splitlines()));rows=[]
for i,q in enumerate(val):
 r0=native[q['id']]['result'];r={'native':r0,'fixedVplus':call(q,[.1,0.]*7)}
 for n in schemes:r[n]={**reuse[n][q['id']]['result'],'reused_validation_checkpoint':selected[n]['best_step']}
 r['validation_selected_structured']={**r[best_name],'selected_family':best_name}
 out={'id':q['id'],'gold':native[q['id']]['gold'],'methods':r,'output_only':outputs(r0)};rows.append(out)
 if (i+1)%50==0:print('FINAL_VALIDATION',i+1,299,flush=True)
(ROOT/'FINAL_VALIDATION_ROWS.json').write_text(json.dumps(rows))
qs=[json.loads(line) for line in (ROOT/'SCIQ_INPUTS.jsonl').read_text().splitlines()];labels={q['id']:q['gold'] for q in map(json.loads,(ROOT/'SCIQ_LABELS.jsonl').read_text().splitlines())};outdir=ROOT/'sciq_results';outdir.mkdir(exist_ok=True)
for i,q in enumerate(qs):
 p=outdir/f'{q["id"]}.json'
 if p.exists():continue
 r0=call(q,[0.]*14);results={'native':r0};cache={('zero',):r0}
 for name,a,kind in [('fixedVplus',[.1,0.]*7,'structured')]+[(n,selected[n]['alpha'],selected[n]['kind']) for n in schemes]:
  key=('zero',) if not any(a) else (kind,*a)
  if key in cache:results[name]={**cache[key],'identical_coefficients_reused':True}
  else:results[name]=call(q,a,kind);cache[key]=results[name]
 results['validation_selected_structured']={**results[best_name],'selected_family':best_name}
 row={'id':q['id'],'gold':labels[q['id']],'methods':results,'output_only':outputs(r0)};p.write_text(json.dumps(row))
 if (i+1)%25==0:print('SCIQ',i+1,len(qs),flush=True)
C.save('EVALUATION_COMPLETE.json',{'status':'COMPLETE','validation_n':len(val),'sciq_n':len(qs),'selected':selected,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
