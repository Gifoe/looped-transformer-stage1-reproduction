import core,json,pathlib
from core import ROOT,torch,bits
import audit_helpers
units=0;checks=0
for p in sorted((ROOT/'data/C').glob('*.json')):
 m=json.loads(p.read_text());t=torch.load(p.with_suffix('.pt'),map_location='cpu',weights_only=False);q=audit_helpers.qs[m['source_reference']['id']]
 for record,key in [(m['native_reference'],'N_logits'),(m['source_reference'],'P_logits')]:audit_helpers.check(t[key],record,q['targets']);checks+=1
 for r in m['records']:audit_helpers.check(t[r['condition']]['logits'],r,q['targets']);checks+=1
 if m['phase']=='technical':assert bits(t['P_logits'],t['zero_attention_identity']['logits'])
 else:units+=1
assert units==512
core.jsave(ROOT/'C_INDEPENDENT_CPU_AUDIT.json',{'status':'PASS','primary_units':units,'full49152_endpoints':checks,'native_source_reference_exact':True,'technical_zero_delta_identity':True,'checks':'full logit hashes, candidate logits, first-tie predictions, ties, FP64 reference logsoftmax vs GPU FP32 tolerance1e-5'})
print('C_CPU_PASS',units,checks)
