"""Fresh unselected technical cases: retain actual D traces, compare CPU independently."""
import bridge as B
from bridge import torch
import json,datetime,hashlib
R=B.ROOT;qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};units=[u for u in map(json.loads,(R/'UNITS.jsonl').read_text().splitlines()) if u['pair_id'] in ['TB000','TB008','TB016'] and not u['reverse_labels'] and u['mapping']==0 and u['direction']==0];m=B.initialize();receipts=[]
for u in units:
 q0=qs[u['reference_W0']];q1=qs[u['input_W1']];r0,z0,c0=B.run(m,q0,capture=True,technical=True);r1,z1,c1=B.run(m,q1,capture=True,technical=True);rr,zz,cc=B.run(m,q1,reference=c0,native=c1,g1=0,mask2=0,capture=True,technical=True);old=json.loads((R/'results'/(u['pair_id']+'.json')).read_text());old=next(x for x in old['units'] if x['id']==u['id']);assert all(rr['scores'][d]['full49152_sha256']==old['conditions']['C00']['scores'][d]['full49152_sha256'] for d in ['1','2']);raw={}
 for kind in ['D_input','D_output']:
  raw[kind]={str(k):{'W0':v.cpu(),'C00':cc[kind][k].cpu()} for k,v in c0[kind].items()}
 raw['S_input']={str(k):{'W1':v.cpu(),'C00':cc['S_input'][k].cpu()} for k,v in c1['S_input'].items()}
 for kind,items in raw.items():
  for k,vs in items.items():a,b=vs.values();assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
 path=B.RAW/(u['pair_id']+'_D_TRACE_WITNESS.pt');torch.save(raw,path);receipts.append({'unit':u['id'],'template':u['template'],'actual_D_layer_inputs_outputs':96,'actual_S_layer_inputs':48,'CPU_bitwise_all_equal':True,'old_A_C00_fullvocabulary_exact':True,'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(R/'LAYER_TRACE_CPU_AUDIT.json').write_text(json.dumps({'status':'PASS','cases':receipts,'additional_body_forwards':9,'scope':'Firstdevbaseperwordingtemplate,frozenindicesbeforetraceoutcomes; actualC00Dinput/outputvsW0andSvsW1CPUtensorcomparisons, no stateoverwrite','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('LAYER_TRACE_CPU_PASS')
