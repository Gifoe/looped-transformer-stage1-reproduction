"""Restore the frozen candidate archive links on a clean reproduction host."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mixed_full'))
from prepare_mixed import link, read, sha

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--stage2',required=True,type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'mixed_full'
registry=read(root/'results/FREEZE.json')
archive=root/'archive'
for directory in ('inputs','scores'):
    (archive/directory).mkdir(parents=True,exist_ok=True)
link(archive/'code',a.stage2/'code')
index={}
for task,name in registry['stage2_zero_shot_archive_sources'].items():
    source=a.stage2 if name=='stage2' else a.stage2/'zero_shot'
    path=source/'inputs'/f'{task}.jsonl'
    if sha(path)!=registry['seven_requests_sha256'][task]:
        raise ValueError(f'Different archived requests for {task}')
    link(archive/'inputs'/path.name,path)
    manifest=path.with_suffix('.jsonl.manifest.json')
    link(archive/'inputs'/manifest.name,manifest)
    link(archive/'scores'/task,source/'scores'/task)
    index[task]=read(source/'COMPARISON.json')[task]
target=archive/'COMPARISON.json'
if target.exists():
    if read(target)!=index:
        raise ValueError('Candidate roster differs')
else:
    temp=target.with_suffix('.partial')
    temp.write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    temp.replace(target)
