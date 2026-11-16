import ATTENTION_EDGE_IMPLEMENTATION as E
from ATTENTION_EDGE_IMPLEMENTATION import torch
import json,datetime
qs={q['id']:q for q in map(json.loads,(E.ROOT/'INPUTS.jsonl').read_text().splitlines())};m=E.initialize();tests=[]
for pid in ['NEW000','NEW016','NEW032']:
 q=qs[pid+'_L0_original'];r,z,c,w=E.forward(m,q,capture=True);old=torch.load(E.RAW/(pid+'.pt'),map_location='cpu',weights_only=False)['native::'+q['id']];ids=torch.tensor([q['ids']],device='cuda')
 with torch.inference_mode():_,states,_=m.model(ids,use_cache=False,past_key_values=None);official={d+1:m.lm_head(states[d])[:,q['R']] for d in range(2)}
 assert all(E.bits(z[d],official[d]) and E.bits(z[d].cpu(),old[d]) for d in [1,2]);tests.append({'id':q['id'],'template':q['template'],'unhooked_observer_formal_full49152_bothpasses_bitwise':True})
(E.ROOT/'FRESH_ZERO_IDENTITY_AUDIT.json').write_text(json.dumps({'status':'PASS','tests':tests,'numerical_audit_body_forwards':6,'scope':'Firstfrozenbasepairperfreshwordingtemplate,selectedwithoutoutcomes; originalunhooked vsobservervsformal native outputs','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('FRESH_NATIVE_AUDIT_PASS')
