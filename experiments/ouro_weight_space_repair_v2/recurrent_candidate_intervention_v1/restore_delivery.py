"""Restore standard lossless records/vector containers; no GPU/private dependencies."""
from pathlib import Path,PurePosixPath
import argparse,json,hashlib,lzma,tarfile
O=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
def safe(n):
 p=PurePosixPath(n);assert n and not p.is_absolute() and '..' not in p.parts and '\\' not in n and ':' not in n
 q=O.joinpath(*p.parts).resolve();assert q.is_relative_to(O.resolve());return q
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args();index=json.loads((O/'DELIVERY_INDEX.json').read_text(encoding='utf8'));parts=json.loads((O/'VECTORS_PARTS_INDEX.json').read_text(encoding='utf8'));h=hashlib.sha256();size=0
 for e in parts['parts']:
  b=safe(e['path']).read_bytes();assert len(b)==e['bytes'] and H(b)==e['sha256'];h.update(b);size+=len(b)
 assert h.hexdigest()==parts['whole_sha256'] and size==parts['whole_bytes']
 packed=safe(parts['whole_name'])
 if packed.exists():assert H(packed.read_bytes())==parts['whole_sha256']
 elif not args.verify_only:
  tmp=packed.with_suffix(packed.suffix+'.restore.tmp');assert not tmp.exists()
  with tmp.open('xb') as out:
   for e in parts['parts']:
    with safe(e['path']).open('rb') as f:
     for block in iter(lambda:f.read(1024*1024),b''):out.write(block)
  assert H(tmp.read_bytes())==parts['whole_sha256'];tmp.rename(packed)
 # Always stream-decode all parts, including verify-only without a combined file.
 dec=lzma.LZMADecompressor();rawhash=hashlib.sha256();rawbytes=0;raw=safe(parts['restored_name']);need_write=not raw.exists() and not args.verify_only;tmp=raw.with_suffix(raw.suffix+'.restore.tmp');out=tmp.open('xb') if need_write else None
 try:
  for e in parts['parts']:
   with safe(e['path']).open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):
     b=dec.decompress(block);rawhash.update(b);rawbytes+=len(b)
     if out:out.write(b)
 finally:
  if out:out.close()
 assert dec.eof and rawbytes==parts['restored_bytes'] and rawhash.hexdigest()==parts['restored_sha256']
 if raw.exists():assert raw.stat().st_size==rawbytes and H(raw.read_bytes())==parts['restored_sha256']
 elif need_write:tmp.rename(raw)
 records=safe(index['records_archive']);assert H(records.read_bytes())==index['records_sha256'];contents=json.loads((O/'RECORDS_CONTENTS.json').read_text(encoding='utf8'));assert H((O/'RECORDS_CONTENTS.json').read_bytes())==index['contents_sha256'];expected={e['path']:e for e in contents['members']};restored=0
 with tarfile.open(records,'r:xz') as t:
  members=t.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
  validated=[]
  for m in members:
   assert m.isfile() and not m.issym() and not m.islnk();p=safe(m.name);b=t.extractfile(m).read();e=expected[m.name];assert len(b)==e['bytes'] and H(b)==e['sha256']
   if p.exists():assert H(p.read_bytes())==e['sha256'],m.name
   validated.append((p,b))
  if not args.verify_only:
   for p,b in validated:
    if p.exists():continue
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b)
    restored+=1
 print(json.dumps(dict(status='lossless_allparts_rawNPZ_records_exact',verify_only=args.verify_only,raw_vectors_bytes=rawbytes,raw_vectors_sha256=rawhash.hexdigest(),record_members=len(expected),restored_members=restored,model_or_GPU_loaded=False)))
if __name__=='__main__':main()
