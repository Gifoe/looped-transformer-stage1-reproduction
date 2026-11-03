"""Stage exact public pinned weights; no model inference or test-set access."""
from pathlib import Path
import hashlib,json,time,requests

ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'models/Ouro-1.4B';DEST.mkdir(parents=True,exist_ok=True)
SPEC=json.loads((ROOT/'sources/ouro_hub.json').read_text(encoding='utf8'))
STATE={};START=time.time()
def record():
 p=ROOT/'logs/MODEL_LOCAL_DOWNLOAD.json';tmp=p.with_suffix('.tmp')
 tmp.write_text(json.dumps(dict(updated_unix=time.time(),elapsed_seconds=time.time()-START,files=STATE),indent=2),encoding='utf8');tmp.replace(p)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify(p,item):
 assert p.stat().st_size==item['size']
 expected=item.get('lfs',{}).get('sha256')
 if expected:assert digest(p)==expected
 elif item.get('blobId'):
  payload=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==item['blobId']
for item in SPEC['siblings']:
 name=item['rfilename']
 if name.startswith('assets/') or name in ['README.md','.gitattributes']:continue
 p=DEST/name;partial=p.with_suffix(p.suffix+'.download');expected=item.get('lfs',{}).get('sha256')
 if p.exists():
  verify(p,item)
  STATE[name]=dict(status='verified',bytes=p.stat().st_size);record();continue
 for attempt in range(5):
  try:
   s=requests.Session();s.trust_env=False;offset=partial.stat().st_size if partial.exists() else 0
   q=s.get('https://hf-mirror.com/ByteDance/Ouro-1.4B/resolve/'+SPEC['sha']+'/'+name,
       headers={'Range':f'bytes={offset}-','Accept-Encoding':'identity'} if offset else {'Accept-Encoding':'identity'},stream=True,timeout=(15,90))
   q.raise_for_status();mode='ab' if offset and q.status_code==206 else 'wb'
   if mode=='wb':offset=0
   with partial.open(mode) as f:
    for b in q.iter_content(4*1024*1024):
     if b:f.write(b);offset+=len(b)
     STATE[name]=dict(status='downloading',completed_bytes=offset,bytes=item['size']);record()
   verify(partial,item)
   partial.replace(p);STATE[name]=dict(status='verified',bytes=p.stat().st_size,sha256=digest(p));record();print(name,'verified',flush=True);break
  except Exception as e:
   STATE[name]=dict(status='retry',attempt=attempt+1,error_type=type(e).__name__);record()
   if attempt==4:raise RuntimeError('Download failed for '+name+'; original partial retained') from None
   time.sleep(2)
metadata=dict(repo='ByteDance/Ouro-1.4B',sha=SPEC['sha'])
if (DEST/'snapshot_metadata.json').exists():assert json.loads((DEST/'snapshot_metadata.json').read_text())==metadata
else:(DEST/'snapshot_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
certificate=dict(status='passed',revision=SPEC['sha'],source='Pinned HF LFS SHA256 / Git blob verification',files=STATE)
receipt=ROOT/'logs/MODEL_SERVER_VERIFIED.json'
if not receipt.exists():receipt.write_text(json.dumps(certificate,indent=2),encoding='utf8')
else:assert json.loads(receipt.read_text())['status']=='passed' and json.loads(receipt.read_text())['revision']==SPEC['sha']
print('ALL_PINNED_MODEL_FILES_VERIFIED',flush=True)
