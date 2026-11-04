"""R24 CPU namespace and immutable source/input/public gates."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];LOCK='configs/r24_MEMORY_SUPPORT_LOCK.json'
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def save(p,x):
 assert Path(p).name.startswith('r24_')
 with (ROOT/p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def gate():
 s=read(LOCK);assert s['status']=='frozen_CPU_memory_support'
 for p,h in s['source_sha256'].items():assert sha('scripts/'+p)==h
 for p,h in s['input_sha256'].items():assert sha(p)==h
 r=read('logs/r24_PUBLIC_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s
