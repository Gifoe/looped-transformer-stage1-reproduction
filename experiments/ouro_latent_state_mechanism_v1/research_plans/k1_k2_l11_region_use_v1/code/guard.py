import pathlib,json,hashlib,sys
R=pathlib.Path('/root/ouro-region-use-sol-20261117');D=pathlib.Path('/root/ouro-binding-interface-sol-20261117');P=pathlib.Path('/root/ouro-semantic-read-sol-20261117')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def verify():
 f=R/'REGION_FREEZE.json';z=json.loads(f.read_text())
 for n,s in z['files_sha256'].items():assert sha(R/n)==s,('sourcechanged',n)
 assert sha(D/'DISCOVERY_FREEZE.json')==z['discovery_freeze_sha256'];assert sha(P/'SCIENCE_FREEZE.json')==z['parent_freeze_sha256']
 for n,s in z['inherited_files_sha256'].items():assert sha(pathlib.Path(n))==s,('inheritedchanged',n)
 return sha(f)
def exact(rows,expected):
 a=[x['id'] for x in rows];assert len(a)==len(set(a))==len(expected) and set(a)==set(expected)
