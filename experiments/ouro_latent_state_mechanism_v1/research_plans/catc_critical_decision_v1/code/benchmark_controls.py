import CATC_CRITICAL as C
from CATC_CRITICAL import torch
import pathlib,json,time,datetime
R=C.ROOT;O=pathlib.Path('/root/ouro-catc-sol-20261116');s=json.loads((R/'SELECTED_METHOD.json').read_text());f=json.loads((O/'FREEZE.json').read_text());qs={q['id']:q for q in map(json.loads,(O/'INPUTS.jsonl').read_text().splitlines())};m=C.initialize();rows=[]
for id in f['technical_ids']:
 q={**qs[id]};old=json.loads((O/'results'/f'{id}.json').read_text());target=next(z for z in old['records'] if z['name']==s['original_record']);q['_dose_targets']=[v[3] for v in target['norms']]
 for kind in ['matched_random','matched_scaling']:
  r,z,rec=C.forward(m,q,*s['coefficients'],kind=kind);assert len(r['dose_matches'])==7 and all(z['actual_norm']<=z['cap'] for z in r['dose_matches']);rows.append({'id':id,'kind':kind,'record':r})
C.save('CONTROL_TECHNICAL_BENCHMARK.json',{'status':'PASS','rows':rows,'estimated_512question_seconds':sum(q['record']['seconds'] for q in rows)/3*512,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print('CONTROL_BENCHMARK_PASS',sum(q['record']['seconds'] for q in rows)/3*512)
