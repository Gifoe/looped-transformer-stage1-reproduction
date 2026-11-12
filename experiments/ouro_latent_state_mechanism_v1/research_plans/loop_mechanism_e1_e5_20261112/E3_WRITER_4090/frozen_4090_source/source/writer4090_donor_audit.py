"""Independent deterministic frozen donor audit, without effect labels."""
import json,hashlib,pathlib,torch
from safetensors.torch import load_file
R=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113/results');L=R/'frozen_unrelated_4090.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def record(base,c):
 stem=hashlib.sha256(c['id'].encode()).hexdigest();m=json.loads((base/(stem+'.json')).read_text());t=load_file(str(base/(stem+'.safetensors')));return m,t
l=json.loads(L.read_text());tr=json.loads((R/'reference_train64/RECEIPT.json').read_text());te=json.loads((R/'technical_reference_three/RECEIPT.json').read_text());assert l['train_reference_receipt_sha256']==sha(R/'reference_train64/RECEIPT.json') and l['technical_receipt_sha256']==sha(R/'technical_reference_three/RECEIPT.json');train={c['id']:record(R/'reference_train64',c) for c in tr['cases']};cases=[(c,record(R/'reference_train64',c)) for c in tr['cases']]+[(c,record(R/'technical_reference_three',c)) for c in te['cases']];checks=0
for c,(m,t) in cases:
 candidates=sorted((i for i,(dm,dt) in train.items() if i!=c['id'] and dm['input_ids_sha256']!=m['input_ids_sha256']),key=lambda i:hashlib.sha256(('q-to-r-interval-v1|'+c['id']+'|'+i).encode()).digest())
 for j in range(8,16):
  for k in ('a','b'):
   tag=f'layer{j:02d}_{k}_delta';d=t[tag];row=l['mapping'][c['id']]['sites'][f'{j:02d}_{k}'];assert row['reference_sha256']==digest(d)
   if d.norm()==0:assert row['status']=='ZERO_REFERENCE_IDENTITY' and row['donor_id'] is None
   else:
    found=next((i for i in candidates if train[i][1][tag].norm()>0),None)
    if found is None:assert row['status']=='UNDEFINED_NO_NONZERO_DONOR'
    else:
     donor=train[found][1][tag];u=donor*(d.norm()/donor.norm());assert row['status']=='OK' and row['donor_id']==found and row['donor_delta_sha256']==digest(donor) and row['unrelated_sha256']==digest(u) and abs(float(u.norm()-d.norm()))<=1e-10*max(1,float(d.norm()))
   checks+=1
out=R/'frozen_unrelated_4090_independent_audit.json';assert not out.exists();out.write_text(json.dumps({'status':'4090_DONOR_MAPPING_INDEPENDENT_AUDIT_PASSED','ledger_sha256':sha(L),'site_checks':checks,'script_sha256':sha(pathlib.Path(__file__)),'scope':'Independent CPU donor ranking, exclusion, dose and vector hashes; no treatment effects.'},indent=2)+'\n');print(sha(out))
