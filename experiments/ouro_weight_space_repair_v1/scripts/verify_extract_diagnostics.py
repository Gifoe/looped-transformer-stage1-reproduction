"""Verify a closed evidence archive and mirror it without replacing different files."""
import hashlib,json,tarfile
from runtime import ROOT,save

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 manifest=json.loads((ROOT/'artifacts/ACCEPTED_DIAGNOSTICS_MANIFEST.json').read_text());archive=ROOT/'artifacts/accepted_diagnostics.tar.gz'
 assert digest(archive)==manifest['archive_sha256']
 files={r['path']:r for r in manifest['files']};seen=set()
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   assert member.isfile() and member.name in files
   target=(ROOT/member.name).resolve();assert target.is_relative_to(ROOT.resolve())
   expected=files[member.name]
   if target.exists():assert target.stat().st_size==expected['bytes'] and digest(target)==expected['sha256'],'Would replace different existing evidence: '+member.name
   else:
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as source,target.open('xb') as out:
     for block in iter(lambda:source.read(8*1024*1024),b''):out.write(block)
    assert target.stat().st_size==expected['bytes'] and digest(target)==expected['sha256']
   seen.add(member.name)
 assert seen==set(files)
 save('artifacts/LOCAL_DIAGNOSTICS_VERIFIED.json',dict(status='passed',files=len(seen),archive_sha256=manifest['archive_sha256'],different_existing_files_overwritten=False))
 print('diagnostics_verified',len(seen),'files')
if __name__=='__main__':main()
