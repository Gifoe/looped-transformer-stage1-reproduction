"""Apply the accepted manifest: this gated run deliberately has no weight edit."""
import argparse,hashlib,json
from pathlib import Path
from runtime import ROOT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=ROOT/'weight_edit_manifest.json');a=ap.parse_args()
 m=json.loads(a.manifest.read_text())
 if m['actual_changed_parameters']!=0 or m['edits']:
  raise RuntimeError('No nonzero edit was accepted in this study; refusing an unvalidated patch')
 assert m['primary_weight_edit'] is None
 p=ROOT/'models/Ouro-1.4B/model.safetensors';h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==m['base_model_sha256'],'Base model differs from the evaluated checkpoint'
 print(json.dumps(dict(status='no_edit',changed_parameters=0,model_path=str(p.parent),base_sha256=h.hexdigest())))
if __name__=='__main__':main()
