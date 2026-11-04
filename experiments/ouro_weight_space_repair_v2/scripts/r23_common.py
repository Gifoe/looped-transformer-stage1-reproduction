"""R23 namespace source/input/public gate, no heldout authorization."""
import json
from r7_core import ROOT,read,sha,digest,jsonlines
LOCK='configs/r23_SLED_PILOT_LOCK.json'
def save(path,x):
 p=ROOT/path;assert p.name.startswith('r23_');tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def gate():
 s=read(LOCK);assert s['status']=='frozen_bounded_development_pilot' and len(s['cohort'])==24 and not s['heldout_access_authorized']
 for p,h in s['source_sha256'].items():assert sha('scripts/'+p)==h
 for p,h in s['inherited_source_sha256'].items():assert sha('scripts/'+p)==h
 for p,h in s['input_sha256'].items():assert sha(p)==h
 r=read('logs/r23_PUBLIC_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s
