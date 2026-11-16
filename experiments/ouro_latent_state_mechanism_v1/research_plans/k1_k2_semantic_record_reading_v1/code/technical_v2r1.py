import record_operator_v2r1 as O
from record_operator_v2r1 import torch
import pathlib,json,datetime,time
R=O.ROOT;P=pathlib.Path('/root/ouro-temporal-bridge-sol-20261117');old=json.loads((R/'TECHNICAL_NUMERICAL_GATE.json').read_text());assert old['status']=='FAIL_PERMUTATION_NULL';m=O.initialize();records=[];start=time.monotonic()
for n,previous in enumerate(old['records']):
 q=previous['q'];arm=previous['arm'];native,nh,pref,_=O.run(m,q,capture_prefix=True);assert all(native['scores'][d]['full49152_sha256']==previous['conditions']['native']['scores'][d]['full49152_sha256'] for d in ['1','2']);group={'native':native};heads={'native':{k:v.cpu() for k,v in nh.items()}};witnesses={}
 for name in ['noop','self','semantic','V','joint','random']:
  target={str(e['layer']):e['actual_delta_norms'] for e in group['semantic']['audit']['events']} if name=='random' else None
  z,h,_,w=O.run(m,q,kind=name,arm=arm,native_prefix=pref,witness=n in [0,16,23,24],target_doses=target);group[name]=z;heads[name]={k:v.cpu() for k,v in h.items()}
  if name in ['noop','self','joint']:assert all(O.bits(nh[k],h[k]) for k in [1,2])
  if name=='V':assert all(O.bits(heads['semantic'][k],h[k].cpu()) for k in [1,2])
  if w:witnesses[name]=w
 torch.save(heads,O.RAW/('v2r1_technical_'+q['id']+'.pt'))
 if witnesses:torch.save({'q':q,'arm':arm,'witnesses':witnesses},O.RAW/('v2r1_witness_'+q['id']+'.pt'))
 records.append({'id':q['id'],'arm':arm,'conditions':group});print('V2_TECHNICAL_WORLD',n+1,32,flush=True)
out={'status':'GPU_PASS_PENDING_CPU_ORACLE','forwards':224,'worlds':32,'records':records,'v1_failed_preserved':True,'operator_estimand':'BF16(actualnativeSDPA + idealFP64pairedmessageDelta), not literalnativepermutedKV SDPA','identity_scope':'self/jointdelta0 andK/Vshareddelta areconstruction identities, notscientificcontrols','wall_seconds':time.monotonic()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'V2R1_TECHNICAL_GPU.json').write_text(json.dumps(out,indent=2));print('V2_GPU_NUMERICAL_PASS_PENDING_CPU',out['wall_seconds'])
