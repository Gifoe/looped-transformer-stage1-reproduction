"""R33 source gates and separate outputs; old frozen helpers remain unchanged."""
import json
from r28_common import ROOT,read,sha,digest,Vectors,PositionBuffers,assert_fresh
LOCK='configs/r33_COMMON_GAIN_PILOT_LOCK.json'
OUTPUTS=['results/r33_'+n for n in ['INPUT_AUDIT.json','NATIVE_PARITY.json','PROGRESS.json','SCORES.json','COMPLETE.json','EXECUTION_AUDIT.json','REPRESENTATIVE_VECTORS.npz','LABEL_FREE_INTEGRITY.json','ANALYSIS.json']]+['logs/r33_'+n for n in ['SUPERVISOR.json','gpu_budget.json','worker.out','worker.err']]
def save(p,x):
 p=ROOT/p;assert p.name.startswith('r33_');tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def pack_diag(diag):
 import torch
 return {k:(v.detach().cpu().item() if isinstance(v,torch.Tensor) else v.item() if hasattr(v,'item') else v) for k,v in diag.items()}
def gate():
 s=read(LOCK);assert s['status']=='frozen_bounded_development_pilot'
 for group in ['source_sha256','input_sha256']:
  for p,h in s[group].items():assert '\\' not in p and sha(p)==h,p
 r=read('logs/r33_PUBLIC_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s
