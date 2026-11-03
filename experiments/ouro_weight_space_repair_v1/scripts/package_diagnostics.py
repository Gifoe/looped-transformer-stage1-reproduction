"""Package only completed accepted diagnostic evidence, while frozen eval runs."""
import hashlib,json,tarfile,time
from runtime import ROOT,save

def main():
 folders=['results/original_call_parity','results/discovery_original','results/layer_screen_original','results/shared_confirmation_original']
 for folder in folders:
  name='COMPLETE_192.json' if folder.endswith('discovery_original') else 'COMPLETE.json'
  assert json.loads((ROOT/folder/name).read_text())['status']=='complete'
 paths=[]
 for folder in folders:paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
 paths.extend(ROOT/p for p in ['data/discovery.jsonl','data/selection.jsonl','data/fixed_demonstrations.json','data/shared_demo_prefix.txt','data/TRAIN_MANIFEST_AUDIT.json','configs/representative_cases.json'])
 paths=sorted(set(paths));out=ROOT/'artifacts/accepted_diagnostics.tar.gz';out.parent.mkdir(exist_ok=True)
 assert not out.exists();temporary=out.with_suffix('.gz.partial');start=time.time();manifest=[]
 with tarfile.open(temporary,'w:gz',compresslevel=1) as archive:
  for p in paths:
   assert p.resolve().is_relative_to(ROOT.resolve())
   archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
   manifest.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 temporary.replace(out)
 h=hashlib.sha256()
 with out.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 save('artifacts/ACCEPTED_DIAGNOSTICS_MANIFEST.json',dict(status='complete',archive_sha256=h.hexdigest(),archive_bytes=out.stat().st_size,files=manifest,elapsed_seconds=time.time()-start,contains_test=False,contains_checkpoint_weights=False))
 print('accepted_diagnostics_packaged',out.stat().st_size,len(paths),flush=True)
if __name__=='__main__':main()
