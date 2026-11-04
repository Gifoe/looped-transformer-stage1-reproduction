"""R25 immutable namespace/public gates."""
import json
from r7_core import ROOT,read,sha,digest,jsonlines
LOCK='configs/r25_HELLINGER_MEMORY_LOCK.json'
def save(p,x):
 q=ROOT/p;assert q.name.startswith('r25_');tmp=q.with_suffix(q.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(q)
def gate():
 s=read(LOCK);assert s['status']=='frozen_bounded_development_pilot' and len(s['cohort'])==24 and not s['heldout_access_authorized']
 for p,h in s['source_sha256'].items():assert sha('scripts/'+p)==h
 for p,h in s['inherited_source_sha256'].items():assert sha('scripts/'+p)==h
 for p,h in s['input_sha256'].items():assert sha(p)==h
 r=read('logs/r25_PUBLIC_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s
