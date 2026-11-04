"""R22 paths/gates; gold-free worker consumes only sanitized frozen manifest."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];LOCK='configs/r22_TRAJECTORY_LOCK.json'
def sha(p):
 h=hashlib.sha256()
 with (ROOT/p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def save(p,x):
 assert Path(p).name.startswith('r22_');q=ROOT/p
 with q.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def gate():
 s=read(LOCK);assert s['status']=='frozen_CPU_trajectory_diagnostic'
 for p,h in s['source_sha256'].items():assert sha('scripts/'+p)==h
 for p,h in s['input_sha256'].items():assert sha(p)==h
 r=read('logs/r22_PUBLIC_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s
