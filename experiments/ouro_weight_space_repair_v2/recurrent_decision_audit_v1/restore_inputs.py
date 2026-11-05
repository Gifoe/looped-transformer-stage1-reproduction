"""Restore only missing inherited evidence into this audit directory, never old paths."""
from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,io
ROOT=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def restore():
 reports=[]
 for idxpath,contents,wanted in [('artifacts/r7_RECORDS_INDEX.json','artifacts/r7_RECORDS_CONTENTS.json',{'results/r7_test_final.jsonl'}),('artifacts/r10_r11_DEVELOPMENT_INDEX.json',None,{'results/r10_development_initial.jsonl','results/r10_development_initial_COMPLETE.json'})]:
  idx=load(idxpath);a=idx.get('archive',{});path=a.get('path',idx.get('archive_path'));h=a.get('sha256',idx.get('archive_sha256'))
  if idx.get('parts'):
   parts=[]
   for part in idx['parts']:
    b=(ROOT/part['path']).read_bytes();assert sha(b)==part['sha256'] and len(b)==part['bytes'];parts.append(b)
   raw=b''.join(parts)
  else:raw=(ROOT/path).read_bytes()
  assert sha(raw)==h
  if contents:
   assert sha((ROOT/contents).read_bytes())==idx['contents_sha256'];entries=load(contents)['files']
  else:entries=idx['files']
  entries={r['path']:r for r in entries}
  with tarfile.open(fileobj=io.BytesIO(raw),mode='r:xz') as t:
   for m in t:
    if m.name not in wanted:continue
    assert m.isfile() and not m.issym() and '..' not in PurePosixPath(m.name).parts
    b=t.extractfile(m).read();e=entries[m.name];assert sha(b)==e['sha256'] and len(b)==e['bytes'];p=OUT/'recovered'/m.name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==b
    else:p.write_bytes(b)
    reports.append(dict(original_path=m.name,recovered_path=p.relative_to(OUT).as_posix(),sha256=sha(b),archive=path,archive_sha256=h))
 (OUT/'inherited_recovery_receipt.json').write_text(json.dumps(dict(status='verified_isolated_recovery',members=reports,old_files_modified=False),indent=2)+'\n',encoding='utf8')
 return reports
if __name__=='__main__':print(json.dumps(restore(),indent=2))
