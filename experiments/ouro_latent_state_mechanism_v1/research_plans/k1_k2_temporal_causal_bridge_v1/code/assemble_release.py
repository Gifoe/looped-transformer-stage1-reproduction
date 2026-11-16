"""Restore lossless large scalar archives from GitHub-sized shards."""
import pathlib,json,hashlib

def assemble(root):
 root=pathlib.Path(root);mp=root/'SHARDED_ARTIFACTS.json'
 if not mp.exists():return
 for spec in json.loads(mp.read_text())['artifacts']:
  dest=root/spec['path'];dest.parent.mkdir(exist_ok=True,parents=True)
  if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==spec['sha256']:continue
  data=b''.join((root/p).read_bytes() for p in spec['parts']);assert len(data)==spec['bytes'] and hashlib.sha256(data).hexdigest()==spec['sha256'];dest.write_bytes(data)
if __name__=='__main__':assemble(pathlib.Path(__file__).resolve().parent.parent)
