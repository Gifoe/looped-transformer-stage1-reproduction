import INTERVENTION_IMPLEMENTATION as I
from INTERVENTION_IMPLEMENTATION import torch
import json,pathlib,datetime
R=I.ROOT;O=pathlib.Path('/root/ouro-catc-sol-20261116');qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};f=json.loads((O/'FREEZE.json').read_text());spec=json.loads((R/'FREEZE.json').read_text());m=I.initialize();records=[]
for id in f['technical_ids']:
 q=qs[id];old=torch.load(O/'results'/f'{id}.pt',map_location='cpu',weights_only=False)['T0'];ids=torch.tensor([q['ids']],device='cuda')
 with torch.inference_mode():_,states,_=m.model(ids,use_cache=False,past_key_values=None);official={d+1:m.lm_head(states[d])[:,q['R']].cpu() for d in range(4)}
 r,z,obs=I.forward(m,q);assert all(torch.equal(z[d].view(torch.uint8),official[d].view(torch.uint8)) and torch.equal(z[d].view(torch.uint8),old[d].view(torch.uint8)) for d in range(1,5));records.append({'id':id,'condition':'N','record':r,'official_and_previous_all4_bitwise':True})
 for name,(region,eta) in spec['conditions'].items():
  if name=='N':continue
  rr,zz,oo=I.forward(m,q,region,eta);assert torch.equal(zz[1].view(torch.uint8),z[1].view(torch.uint8));assert len(oo)==21;records.append({'id':id,'condition':name,'record':rr})
 (R/'technical'/f'{id}.pt').parent.mkdir(exist_ok=True);torch.save({'native_full_heads':z,'native_attention_observations':obs},R/'technical'/f'{id}.pt')
(R/'TECHNICAL_PASS.json').write_text(json.dumps({'status':'PASS','records':records,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('CONTEXT_TECHNICAL_PASS')
