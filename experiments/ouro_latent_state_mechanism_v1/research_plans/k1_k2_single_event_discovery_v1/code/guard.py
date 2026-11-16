import pathlib,json,hashlib,sys
ROOT=pathlib.Path('/root/ouro-binding-interface-sol-20261117');PARENT=pathlib.Path('/root/ouro-semantic-read-sol-20261117')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify():
 f=ROOT/'DISCOVERY_FREEZE.json';z=json.loads(f.read_text())
 for path,digest in z['files_sha256'].items():assert sha(ROOT/path)==digest,('newsourcechanged',path)
 assert sha(PARENT/'SCIENCE_FREEZE.json')==z['parent_science_freeze_sha256'];sys.path.insert(0,str(PARENT/'code'));import freeze_guard;assert freeze_guard.verify()==z['parent_science_freeze_sha256']
 for name,digest in z['parent_native_json_sha256'].items():assert sha(PARENT/'native_results'/name)==digest
 return sha(f)
def exact_ids(rows,expected):
 a=[u['id'] for u in rows];assert len(a)==len(set(a))==len(expected) and set(a)==set(expected)
