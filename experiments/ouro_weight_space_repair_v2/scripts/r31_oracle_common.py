"""CPU diagnostic namespace, immutable inputs and single-execution outputs."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
LOCK='configs/r31_SHARED_PREFIX_ORACLE_LOCK.json'
OUTPUTS=['results/r31_ORACLE_'+n for n in ['RAW.json','COMPLETE.json','ANALYSIS.json','PROGRESS.json']]+['logs/r31_ORACLE_'+n for n in ['SUPERVISOR.json','worker.out','worker.err','analyzer.out','analyzer.err']]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,x,replace=False):
 p=ROOT/p;assert p.name.startswith('r31_ORACLE_');assert replace or not p.exists();tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
 tmp.replace(p)
def gate():
 s=read(LOCK);assert s['status']=='frozen_CPU_shared_prefix_capacity_oracle'
 for group in ['source_sha256','input_sha256']:
  for p,h in s[group].items():assert '\\' not in p and sha(p)==h,p
 r=read('logs/r31_ORACLE_PUBLIC_DEFINITION_RECEIPT.json');assert r['readback_verified'] and r['protocol_sha256']==sha(LOCK)
 return s
