"""Audit-only repair: original serializer used colliding world/unit IDs.
Recompute frozen native endpoints, require exact stored full-logit SHA, save missing B.
No intervention rerun, changed scientific criterion or sample selection.
"""
import core,json,time,pathlib,datetime
from core import ROOT,torch,cpu,bits,forward,tsave,jsave
while not (ROOT/'B_heldout_COMPLETE.json').exists():time.sleep(5)
model,lock=core.initialize(worker='B');qs={q['id']:q for q in map(json.loads,(ROOT/'B_INPUTS.jsonl').read_text().splitlines())};out=ROOT/'data/B_native_restored';out.mkdir(exist_ok=True);tests=[]
for p in sorted((ROOT/'data/B').glob('*.json')):
 m=json.loads(p.read_text());saved={};entry=[]
 for u in m['units']:
  q=qs[u['receiver']];z1,z2,n=forward(model,q,mode='native');r={x['condition']:x for x in u['records']}
  assert core.digest(z1)==r['K1']['logits_sha256'] and core.digest(z2)==r['native']['logits_sha256'],'recovered endpoint must match original bytes'
  saved[q['id']]={'B':cpu(n['B']),'K1_logits':cpu(z1),'native_logits':cpu(z2),'stages':{k:cpu(v) for k,v in n['stages'].items()}};entry.append({'id':q['id'],'K1_full49152_sha256_exact':True,'K2_full49152_sha256_exact':True,'B_sha256':core.digest(n['B']),'seconds':n['elapsed_seconds']})
 tsave(out/f'{m["base_pair_id"]}.pt',saved);tests.extend(entry);print(m['base_pair_id'],'native_recovered_exact',flush=True)
jsave(ROOT/'B_NATIVE_SERIALIZATION_REPAIR.json',{'status':'PASS','cause':'Native world IDs and intervention unit IDs had identical dictionary keys; unit patch dictionary overwrote native original/donor entries during serialization only. Recorded native scalar endpoints and running donor/receiver B used by interventions were unaffected.','resolution':'Freeze-preserving audit-only native recollection; every full49152 K1/K2 hash must match original science records before saving the missing boundary state. Original tensor files preserved. Recovered B in data/B_native_restored, edits remain in data/B.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests':tests})
