"""Restore byte-exact per-question JSON from the public lossless archives."""
import argparse,gzip,json,hashlib
from pathlib import Path
a=argparse.ArgumentParser();a.add_argument('--release',default='.');a.add_argument('--output',required=True);args=a.parse_args();root=Path(args.release);out=Path(args.output).resolve();n=0
for archive in sorted((root/'results/raw_jsonl').glob('*.jsonl.gz')):
 with gzip.open(archive,'rt',encoding='utf8') as f:
  for line in f:
   item=json.loads(line);relative=Path(item['relative_path']);assert not relative.is_absolute() and '..' not in relative.parts;data=item['json'].encode('utf8');assert hashlib.sha256(data).hexdigest()==item['sha256'];target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():assert target.read_bytes()==data
   else:target.write_bytes(data)
   n+=1
print(json.dumps({'status':'ALL_SMALL_RAW_JSON_RESTORED_EXACTLY','files':n,'output':str(out)}))
