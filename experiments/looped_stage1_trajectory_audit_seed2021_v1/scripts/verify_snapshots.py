"""Verify every checkpoint byte against pinned public LFS SHA-256 records."""
import hashlib
import json
from pathlib import Path
import requests
import time

ROOT=Path(__file__).resolve().parents[1]
session=requests.Session();session.trust_env=False;records=[]
for repo in ['ByteDance/Ouro-1.4B','Qwen/Qwen3-4B-Base']:
    directory=ROOT/'models'/repo.split('/')[-1]
    revision=json.loads((directory/'snapshot_metadata.json').read_text())['sha']
    response=session.get(f'https://hf-mirror.com/api/models/{repo}/revision/{revision}',params={'blobs':'true'},timeout=30)
    response.raise_for_status();meta=response.json()
    assert meta['sha']==revision
    for entry in meta['siblings']:
        name=entry['rfilename'];path=directory/name
        if not path.is_file():continue
        start=time.monotonic();actual=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        expected=entry.get('lfs',{}).get('sha256')
        if expected:assert actual==expected,(repo,name,actual,expected)
        records.append(dict(repo=repo,revision=revision,file=name,bytes=path.stat().st_size,sha256=actual,expected_lfs_sha256=expected,verified_against_lfs=bool(expected),elapsed_s=time.monotonic()-start))
        print(json.dumps(records[-1]),flush=True)
(ROOT/'results/checkpoint_integrity.json').write_text(json.dumps(dict(passed=True,files=records),indent=2),encoding='utf-8')
