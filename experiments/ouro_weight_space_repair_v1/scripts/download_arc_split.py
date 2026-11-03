"""Pinned public parquet retrieval, with mandatory candidate/test lock."""
import argparse,hashlib,json,time,urllib.request
from runtime import ROOT,save,configure

REV='210d026faf9955653af8916fad021475a3f00453'
def main():
 configure();ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','validation','test'],required=True);a=ap.parse_args()
 lockpath=None
 if a.split!='train':
  lockpath=ROOT/'configs'/('FINAL_LOCK.json' if a.split=='test' else 'CANDIDATE_LOCK.json')
  lock=json.loads(lockpath.read_text());assert lock['candidate_definitions_frozen'] is True
  if a.split=='test':assert lock['test_evaluation_authorized'] is True
 target=ROOT/'data/raw'/f'{a.split}.parquet'
 target.parent.mkdir(parents=True,exist_ok=True)
 metadata=json.loads((ROOT/'sources/arc_lfs_metadata.json').read_text())
 item=next(x for x in metadata if x['path']==f'ARC-Challenge/{a.split}-00000-of-00001.parquet')
 def verify(payload):
  assert len(payload)==item['size'] and hashlib.sha256(payload).hexdigest()==item['lfs']['oid'],'Pinned LFS SHA256 mismatch'
 if target.exists():
  verify(target.read_bytes())
  print(json.dumps(dict(status='already_staged',split=a.split,sha256=hashlib.sha256(target.read_bytes()).hexdigest())));return
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 failures=[]
 for host in ['https://hf-mirror.com','https://huggingface.co']:
  url=f'{host}/datasets/allenai/ai2_arc/resolve/{REV}/ARC-Challenge/{a.split}-00000-of-00001.parquet?download=true'
  try:
   with opener.open(url,timeout=30) as response:payload=response.read()
   assert payload[:4]==payload[-4:]==b'PAR1' and len(payload)>10000
   verify(payload)
   temporary=target.with_suffix('.parquet.partial');temporary.write_bytes(payload);temporary.replace(target)
   save(f'data/{a.split.upper()}_DOWNLOAD.json',dict(status='complete',dataset_revision=REV,split=a.split,url=url,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),lock_sha256=None if lockpath is None else hashlib.sha256(lockpath.read_bytes()).hexdigest(),downloaded_unix=time.time(),prior_failures=failures));print('download_complete',a.split,len(payload));return
  except Exception as exc:failures.append(dict(host=host,error=repr(exc)))
 raise RuntimeError(f'Pinned dataset download failed: {failures}')
if __name__=='__main__':main()
