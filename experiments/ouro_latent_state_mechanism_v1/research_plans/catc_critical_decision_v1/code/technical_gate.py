import CATC_CRITICAL as C
from CATC_CRITICAL import torch
import pathlib,json,time,datetime,hashlib
R=C.ROOT;O=pathlib.Path('/root/ouro-catc-sol-20261116');f=json.loads((O/'FREEZE.json').read_text());qs={q['id']:q for q in map(json.loads,(O/'INPUTS.jsonl').read_text().splitlines())};m=C.initialize();tests=[];witnesses=[]
for id in f['technical_ids']:
 q=qs[id];old=torch.load(O/'results'/f'{id}.pt',map_location='cpu',weights_only=False);r,z,rec=C.forward(m,q)
 assert all(torch.equal(z[k].view(torch.uint8),old['T0'][k].view(torch.uint8)) for k in range(1,5));assert not rec['norms']
 for name,av,ap in [('T1_structured',.1,0),('T2_structured',0,.1),('T3_structured',-.1,0),('T4_structured',0,-.1)]:
  rr,zz,_=C.forward(m,q,av,ap);assert all(torch.equal(zz[k].view(torch.uint8),old[name][k].view(torch.uint8)) for k in [2,4])
 tests.append({'id':id,'zero_all4_full49152_bitwise':True,'four_structured_settings_K2_K4_full49152_bitwise':True,'native_seconds':r['seconds'],'native_peak_bytes':r['peak_allocated_bytes']})
 tech=torch.load(O/'technical'/f'{id}.pt',map_location='cpu',weights_only=False)
 for layer,v in tech['audit_witness'].items():
  p1,p2,v1,v2=[v[n].double() for n in ['p1','p2','v1','v2']];dv=(p2.unsqueeze(-1)*(v2-v1)).sum(-2);dp=((p2-p1).unsqueeze(-1)*v1).sum(-2);sv=(.5*(p1+p2).unsqueeze(-1)*(v2-v1)).sum(-2);sp=(.5*(p2-p1).unsqueeze(-1)*(v1+v2)).sum(-2);total=(p2.unsqueeze(-1)*v2).sum(-2)-(p1.unsqueeze(-1)*v1).sum(-2);cross=((p2-p1).unsqueeze(-1)*(v2-v1)).sum(-2);rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));cos=lambda a,b:float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))
  assert rel(dv+dp,total)<1e-10 and rel(sv+sp,total)<1e-10 and rel(dv-sv,.5*cross)<1e-9 and rel(sp-dp,.5*cross)<1e-9
  witnesses.append({'id':id,'physical_layer':layer+1,'original_reconstruction_relative_error':rel(dv+dp,total),'symmetric_reconstruction_relative_error':rel(sv+sp,total),'original_value_norm':float(dv.norm()),'symmetric_value_norm':float(sv.norm()),'original_routing_norm':float(dp.norm()),'symmetric_routing_norm':float(sp.norm()),'value_cosine':cos(dv,sv),'routing_cosine':cos(dp,sp),'cross_norm':float(cross.norm()),'total_norm':float(total.norm()),'witness_sha256':hashlib.sha256((O/'technical'/f'{id}.pt').read_bytes()).hexdigest()})
C.save('TECHNICAL_GATE.json',{'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_endpoint_tests':tests,'symmetric_audit':witnesses,'interpretation':'Both conventions sum to the same update, reallocating half the interaction term; components are not uniquely identified mechanisms. Actual post-RoPE QKV witnesses and original SDPA numerical reconstruction audit are reused from the same unchanged source/checkpoint/GPU.'});print('CRITICAL_TECHNICAL_GATE_PASS')
