import pathlib,json,hashlib
R=pathlib.Path('/root/ouro-causal-edge-sol-20261117');P=R/'publication';artifacts=json.loads((P/'SHARDED_ARTIFACTS.json').read_text())['artifacts'] if (P/'SHARDED_ARTIFACTS.json').exists() else []
for f in sorted(P.rglob('*')):
 if f.is_file() and f.stat().st_size>2*1024*1024 and f.suffix=='.gz':
  data=f.read_bytes();parts=[]
  for i,o in enumerate(range(0,len(data),2*1024*1024)):
   p=f.with_name(f.name+f'.part{i:03d}');p.write_bytes(data[o:o+2*1024*1024]);parts.append(str(p.relative_to(P)))
  artifacts=[x for x in artifacts if x['path']!=str(f.relative_to(P))];artifacts.append({'path':str(f.relative_to(P)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'parts':parts});f.unlink()
(P/'SHARDED_ARTIFACTS.json').write_text(json.dumps({'artifacts':artifacts,'assembly':'code/reproduce_release.py automatically verifies and reconstructs these lossless archives before scalar replay.'},indent=2))
def manifest():
 virtual={x['path'] for x in artifacts};files=[]
 for f in sorted(P.rglob('*')):
  if not f.is_file() or '__pycache__' in f.parts or f.name=='PUBLICATION_MANIFEST.json' or str(f.relative_to(P)) in virtual:continue
  d=f.read_bytes();files.append({'path':str(f.relative_to(P)),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()})
 (P/'PUBLICATION_MANIFEST.json').write_text(json.dumps({'files':files,'total_bytes':sum(x['bytes'] for x in files),'source_commit':'9b900da9cffc6eba97623a76053bb823f319d464'},indent=2))
manifest();print('SHARDED',len(artifacts))
