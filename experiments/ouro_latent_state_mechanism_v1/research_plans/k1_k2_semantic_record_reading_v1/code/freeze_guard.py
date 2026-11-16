import pathlib,json,hashlib
R=pathlib.Path('/root/ouro-semantic-read-sol-20261117')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def verify():
 f=R/'SCIENCE_FREEZE.json';d=f.read_bytes();z=json.loads(d);identity=hashlib.sha256(d).hexdigest()
 for name,digest in z['source_and_inputs_sha256'].items():assert sha(R/name)==digest,('frozen dependency changed',name)
 from ab1_loader_draft import EXPECTED
 assert z['model_source_sha256']==EXPECTED
 return identity

def exact_ids(rows,expected):
 ids=[u['id'] for u in rows];assert len(ids)==len(set(ids))==len(expected) and set(ids)==set(expected),('ID coverage mismatch',len(ids),len(expected))
