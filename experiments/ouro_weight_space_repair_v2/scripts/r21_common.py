"""R21 immutable CPU diagnostic gates and namespace."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOCK='configs/r21_READOUT_ORACLE_LOCK.json'
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def save(p,x):
 assert Path(p).name.startswith('r21_');q=ROOT/p
 with q.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def gate():
 s=read(LOCK);assert s['status']=='frozen_CPU_readout_oracle'
 for p,h in s['input_sha256'].items():assert sha(p)==h
 for p,h in s['source_sha256'].items():assert sha('scripts/'+p)==h
 r=read('logs/r21_PUBLIC_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s
