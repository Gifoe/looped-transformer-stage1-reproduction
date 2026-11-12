"""Critical frozen-cohort/receipt/precision closure guard after independent numerical audits."""
import sys,json,hashlib
from pathlib import Path
directory,cohort_path,stage,count=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]);c=json.loads(cohort_path.read_text());ids=c['technical_ids'] if count==2 else c['ids'];expected=set(ids);receipt=json.loads((directory/'receipt.json').read_text());identity=json.loads((directory/'execution_identity.json').read_text());rows=[json.loads(p.read_text()) for p in directory.glob('*.results.json')]
assert receipt['status']=='COMPLETE' and receipt['N']==count==len(expected)==len(rows)
assert {r['id'] for r in rows}==expected and len({(r['split'],r['id']) for r in rows})==len(rows)
assert identity['actual_selected_ids_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
ordered=sorted(ids);donors={ident:ordered[(i+1)%len(ordered)] for i,ident in enumerate(ordered)}
for name in (['state_donor_map.json','mediator_donor_map.json'] if stage.startswith('r') else ['control_donor_map.json']):assert json.loads((directory/name).read_text())==donors
precision=[]
if stage!='base':
 precision=sorted(set(c['precision_ids']).intersection(expected));assert identity['precision_ids']==precision
 actual=sorted(r['id'] for r in rows if any('fp32_sdpa_bf16_writeback' in x['condition'] for x in r['rows']));assert actual==precision
result={'status':'FROZEN_SCOPE_AND_PRECISION_COMPLETE','stage':stage,'n':count,'precision_ids':precision,'receipt_sha256':hashlib.sha256((directory/'receipt.json').read_bytes()).hexdigest(),'cohort_sha256':hashlib.sha256(cohort_path.read_bytes()).hexdigest()};(directory/'scope_complete_checked.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
