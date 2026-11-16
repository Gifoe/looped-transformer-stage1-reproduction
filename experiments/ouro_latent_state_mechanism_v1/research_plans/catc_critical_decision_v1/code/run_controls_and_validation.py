import CATC_CRITICAL as C
from CATC_CRITICAL import torch
import pathlib,json,datetime,time,hashlib
R=C.ROOT;O=pathlib.Path('/root/ouro-catc-sol-20261116');f=json.loads((R/'FREEZE.json').read_text());assert json.loads((R/'TECHNICAL_GATE.json').read_text())['status']=='PASS';s=json.loads((R/'SELECTED_METHOD.json').read_text());deadline=datetime.datetime.fromisoformat(f['deadline_utc']).timestamp();qs={q['id']:q for q in map(json.loads,(O/'INPUTS.jsonl').read_text().splitlines())};old={q['id']:q for q in map(json.loads,(O/'TRAINING_FREE_ALL_ROWS.jsonl').read_text().splitlines())};m=C.initialize();a,b=s['coefficients'];start=time.monotonic();done=0
for i,id in enumerate(f['cohorts']['confirmation']):
 p=R/'results'/f'{id}.json'
 if p.exists():continue
 if time.time()>deadline:raise RuntimeError('Hard180minstop: preserve completed questions')
 q={**qs[id]};target=next(v for v in old[id]['records'] if v['name']==s['original_record']);q['_dose_targets']=[v[3] for v in target['norms']];methods={}
 for n in ['matched_random','matched_scaling']:
  r,z,rec=C.forward(m,q,a,b,kind=n);methods[n]=r
 p.write_text(json.dumps({'id':id,'methods':methods,'target_actual_norms':q['_dose_targets'],'input_ids_sha256':q['input_ids_sha256']}));done+=1
 if (i+1)%25==0:print('MATCHED_CONTROLS',i+1,512,flush=True)
(R/'EXPERIMENT2_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','n':len(f['cohorts']['confirmation']),'new_GPU_forwards':2*done,'wall_seconds':time.monotonic()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}));print('MATCHED_CONTROLS_COMPLETE',flush=True)
# All remaining Train endpoints already exist under the same unchanged original protocol.
val=[json.loads(l) for l in (O/'phaseC/VALIDATION_INPUTS.jsonl').read_text().splitlines()];prior={q['id']:q for q in json.loads((O/'phaseC/FINAL_VALIDATION_ROWS.json').read_text())};out=R/'validation';out.mkdir(exist_ok=True);vstart=time.monotonic();new=0
for i,q in enumerate(val):
 p=out/f'{q["id"]}.json'
 if p.exists():continue
 if time.time()>deadline:raise RuntimeError('Hard180minstop: preserve completed validation questions')
 if s['name']=='V+':record=prior[q['id']]['methods']['fixedVplus'];reused=True
 else:record,_,_=C.forward(m,q,a,b);new+=1;reused=False
 p.write_text(json.dumps({'id':q['id'],'gold':prior[q['id']]['gold'],'native':prior[q['id']]['methods']['native'],'selected':record,'output_only':prior[q['id']]['output_only'],'selected_reused':reused,'input_ids_sha256':q['input_ids_sha256']}))
 if (i+1)%50==0:print('VALIDATION',i+1,299,flush=True)
(R/'EXPERIMENT3_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','remaining_Train_n':len(f['cohorts']['remaining_confirmation']),'new_Train_GPU_forwards':0,'validation_n':len(val),'new_Validation_GPU_forwards':new,'validation_wall_seconds':time.monotonic()-vstart,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}));print('EXPERIMENT3_COMPLETE')
