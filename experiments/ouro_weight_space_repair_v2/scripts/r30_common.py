"""R30 distinct outputs/gates; old frozen sources are imported unchanged."""
import json
from r28_common import ROOT,read,sha,digest,Vectors,PositionBuffers,assert_fresh
LOCK='configs/r30_SOURCE_CLOCK_PILOT_LOCK.json'
OUTPUTS=['results/r30_'+n for n in ['INPUT_AUDIT.json','NATIVE_PARITY.json','PROGRESS.json','SCORES.json','COMPLETE.json','EXECUTION_AUDIT.json','REPRESENTATIVE_VECTORS.npz','LABEL_FREE_INTEGRITY.json','ANALYSIS.json']]+['logs/r30_'+n for n in ['SUPERVISOR.json','gpu_budget.json','worker.out','worker.err']]
def save(p,x):
 p=ROOT/p;assert p.name.startswith('r30_');tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def gate():
 s=read(LOCK);assert s['status']=='frozen_bounded_development_pilot'
 for group in ['source_sha256','input_sha256']:
  for p,h in s[group].items():assert '\\' not in p and sha(p)==h,p
 r=read('logs/r30_PUBLIC_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s
