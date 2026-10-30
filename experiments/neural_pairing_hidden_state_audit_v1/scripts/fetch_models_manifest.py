"""Portable download of only the27 pinned files; no credentials in artifacts."""
import hashlib
import json
import os
from linux_runtime import ROOT


def main():
    os.environ['HF_HOME']=str(ROOT/'cache/hf')
    from huggingface_hub import hf_hub_download
    spec=json.loads((ROOT/'configs/model_transfer.json').read_text())
    for rec in spec['files']:
        alias,name=rec['relative_path'].split('/',1)
        p=ROOT/'models'/rec['relative_path'];p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            hf_hub_download(rec['repo'],name,revision=rec['revision'],local_dir=ROOT/'models'/alias)
        assert p.stat().st_size==rec['bytes'],p
        h=hashlib.sha256() if rec.get('sha256') else hashlib.sha1(('blob '+str(rec['bytes'])+'\0').encode())
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==(rec.get('sha256') or rec['git_blob']),p
        print(rec['relative_path'],'verified',flush=True)


if __name__=='__main__':main()
