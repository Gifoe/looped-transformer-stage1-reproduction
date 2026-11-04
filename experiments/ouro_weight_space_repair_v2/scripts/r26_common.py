"""R26 bounded source-gated acceptance; no effect analysis."""
import json
from r7_core import ROOT,read,sha,digest,jsonlines
LOCK='configs/r26_CAUSAL_ACCEPTANCE_LOCK.json'
def save(p,x):
 q=ROOT/p;assert q.name.startswith('r26_');tmp=q.with_suffix(q.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(q)
def gate():
 s=read(LOCK);assert s['status']=='frozen_bounded_acceptance'
 for group in ['source_sha256','input_sha256']:
  for p,h in s[group].items():assert sha(p)==h,p
 r=read('logs/r26_PUBLIC_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s
