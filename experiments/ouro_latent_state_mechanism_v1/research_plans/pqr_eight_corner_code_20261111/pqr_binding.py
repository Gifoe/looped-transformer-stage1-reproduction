"""One-shot binding; do not invoke before all ROUND_2 split coverage completes."""
import argparse,json,hashlib
from pathlib import Path
from pqr_runner import ROOT,EXP,bindings,sha
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve();out.relative_to(ROOT.resolve());b=bindings();longest=None
for split in ('train','validation','test'):
 for line in (EXP/'results'/('INPUT_MANIFEST_'+split+'.jsonl')).open():
  r=json.loads(line)
  if r['candidate']==0:
   key=(r['input_length'],split,r['id'])
   if longest is None or key>longest:longest=key
assert longest
obj={'status':'PQR_BINDING_COMPLETE_PENDING_PROBE','binding':b,'longest_request':{'input_length':longest[0],'split':longest[1],'id':longest[2]}}
with out.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n')
print('BOUND',sha(out),'LONGEST',json.dumps(obj['longest_request']))
