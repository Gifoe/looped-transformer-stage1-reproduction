"""Read-only in-memory publication transport; package allowlist and sizes enforced."""
import argparse,base64,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];P=R/'publication/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_latent_mechanism_completion'
ap=argparse.ArgumentParser();ap.add_argument('action',choices=['catalog','small','chunk']);ap.add_argument('--file');ap.add_argument('--offset',type=int,default=0);ap.add_argument('--length',type=int,default=786432);ap.add_argument('--start',type=int,default=0);a=ap.parse_args()
manifest=json.loads((P/'PACKAGE_MANIFEST.json').read_text());files=list(manifest['files']);data=(P/'PACKAGE_MANIFEST.json').read_bytes();files.append(dict(path='PACKAGE_MANIFEST.json',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),git_blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
text_suffixes={'.py','.md','.json','.jsonl','.txt','.sh','.svg'}
for f in files:f['inline_text']=pathlib.Path(f['path']).suffix in text_suffixes and f['bytes']<=100000
if a.action=='catalog':print(json.dumps(dict(files=files,total_bytes=sum(f['bytes'] for f in files),manifest_sha256=hashlib.sha256(data).hexdigest())));raise SystemExit
if a.action=='small':
 chosen=[];size=0;texts=[f for f in files if f['inline_text']]
 for f in texts[a.start:]:
  if chosen and size+f['bytes']>100000:break
  p=(P/f['path']).resolve();assert p.is_relative_to(P);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==f['sha256'];chosen.append(dict(path=f['path'],content=raw.decode('utf-8'),git_blob_sha=f['git_blob_sha']));size+=len(raw)
 print(json.dumps(dict(start=a.start,next_start=a.start+len(chosen),total=len(texts),files=chosen)));raise SystemExit
f=next(f for f in files if f['path']==a.file);p=(P/a.file).resolve();assert p.is_relative_to(P) and p.stat().st_size==f['bytes'];assert 0<=a.offset<f['bytes'] and 0<a.length<=786432
with p.open('rb') as h:h.seek(a.offset);raw=h.read(a.length)
print(json.dumps(dict(path=a.file,offset=a.offset,bytes=len(raw),total=f['bytes'],base64=base64.b64encode(raw).decode())))
