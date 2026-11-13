"""Package completed E7 exactly, with lossless large-source/result archives."""
import gzip,hashlib,json,shutil,tarfile
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
root=Path('/root/autodl-tmp/ouro-e7/reencoding')
assert read(root/'full_capture/COMPLETE.json')['n']==22280
assert read(root/'analysis/VALIDATION_COMPLETE.json')['status']=='COMPLETE'
assert read(root/'analysis/ROOT_INDEPENDENT_ARITHMETIC_AUDIT.json')['status']=='PASS'
release=Path('/root/autodl-tmp/ouro-e7/E7_release_v1')
assert not release.exists();release.mkdir()
copied=[]
def copy(src,relative):
 dest=release/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
 copied.append({'file':str(relative),'source':str(src),'source_sha256':sha(src),'release_sha256':sha(dest)})
for p in sorted(root.iterdir()):
 if p.is_file() and p.suffix in ['.py','.md','.json']:
  copy(p,Path('source_and_protocol')/p.name)
for p in sorted((root/'analysis').iterdir()):
 if p.is_file():copy(p,Path('analysis')/p.name)
for p in sorted(root.glob('*.jsonl')):
 dest=release/'frozen_inputs'/(p.name+'.gz');dest.parent.mkdir(exist_ok=True)
 with p.open('rb') as src,gzip.open(dest,'wb',compresslevel=6) as dst:shutil.copyfileobj(src,dst,8*1024*1024)
 h=hashlib.sha256()
 with gzip.open(dest,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==sha(p)
 copied.append({'file':str(dest.relative_to(release)),'source':str(p),'uncompressed_sha256':sha(p),'release_sha256':sha(dest)})
for folder in ['full_capture','technical']:
 dest=release/'raw_archives'/(folder+'.tar.gz');dest.parent.mkdir(exist_ok=True);records=[]
 with tarfile.open(dest,'w:gz',compresslevel=6) as tf:
  for p in sorted((root/folder).rglob('*')):
   if not p.is_file():continue
   assert not p.is_symlink() and p.resolve().is_relative_to((root/folder).resolve())
   rel=p.relative_to(root/folder);tf.add(p,arcname=str(rel),recursive=False)
   records.append({'file':str(rel),'sha256':sha(p),'bytes':p.stat().st_size})
 with gzip.open(release/'raw_archives'/(folder+'_FILES.jsonl.gz'),'wt',encoding='utf-8') as f:
  for record in records:f.write(json.dumps(record)+'\n')
 copied.append({'file':str(dest.relative_to(release)),'source':str(root/folder),'release_sha256':sha(dest),'archived_files':len(records)})
for p in sorted(root.glob('*.log')):
 dest=release/'execution_logs'/(p.name+'.gz');dest.parent.mkdir(exist_ok=True)
 with p.open('rb') as src,gzip.open(dest,'wb',compresslevel=6) as dst:shutil.copyfileobj(src,dst)
(release/'RELEASE_SOURCE_MAP.json').write_text(json.dumps(copied,indent=2)+'\n')
files=[{'file':str(p.relative_to(release)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(release.rglob('*')) if p.is_file()]
assert all(p['bytes']<90*1024*1024 for p in files),'Git blob over conservative cap'
(release/'FILE_MANIFEST.json').write_text(json.dumps({'files':files,'count':len(files),'source':'Completed fixed E7 capture/analysis; original raw bytes retained in lossless archives; actual method profile superseded and not run.'},indent=2)+'\n')
archive=Path('/root/autodl-tmp/ouro-e7/E7_PUBLICATION_v1.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=1) as tf:tf.add(release,arcname='E7_natural_reencoding')
print(json.dumps({'status':'PACKAGED','archive':str(archive),'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'file_count':len(files)+1,'largest_blob':max(files,key=lambda x:x['bytes'])},indent=2))
