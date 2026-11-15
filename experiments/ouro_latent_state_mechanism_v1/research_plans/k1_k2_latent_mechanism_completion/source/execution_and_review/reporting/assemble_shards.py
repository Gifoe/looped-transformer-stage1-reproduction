"""Reassemble public byte shards, or verify them without writing files."""
import argparse,hashlib,json,pathlib
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output');ap.add_argument('--verify-only',action='store_true');a=ap.parse_args();r=pathlib.Path(a.root).resolve();m=json.loads((r/'SHARDED_ARTIFACTS.json').read_text());assert a.verify_only or a.output
for relative,item in m.items():
 h=hashlib.sha256();total=0;target=None;handle=None
 if not a.verify_only:
  out=pathlib.Path(a.output).resolve();target=(out/relative).resolve();assert target.is_relative_to(out);target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();handle=target.with_suffix(target.suffix+'.partial').open('xb')
 for part in item['parts']:
  path=(r/part['path']).resolve();assert path.is_relative_to(r);data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==part['sha256'];h.update(data);total+=len(data)
  if handle:handle.write(data)
 if handle:handle.close()
 assert total==item['bytes'] and h.hexdigest()==item['sha256'],relative
 if target:target.with_suffix(target.suffix+'.partial').rename(target)
print(json.dumps(dict(status='PASS',n_sharded_artifacts=len(m),scope='Byte-exact archival verification/reassembly, not model inference.')))
