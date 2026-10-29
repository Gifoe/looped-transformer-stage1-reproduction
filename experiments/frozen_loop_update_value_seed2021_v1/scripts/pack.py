"""Compact transfer bundle, excluding caches, checkpoints and transport credentials."""
from pathlib import Path
import zipfile,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
name=sys.argv[1]
assert name.replace('_','').isalnum()
out=ROOT/'logs'/(name+'.zip');files=[]
for folder in ['configs','scripts','results','data','sources','logs']:
 for p in (ROOT/folder).glob('*'):
  if p.is_file() and p.suffix in ['.json','.jsonl','.py','.ps1','.yaml','.out','.err','.log','.md'] and p.name!='runtime_probe.py':files.append(p)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,str(p.relative_to(ROOT)))
print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),files=len(files))))
