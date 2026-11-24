"""Retrieve the two pinned public ScienceQA files used for this experiment."""

from hashlib import file_digest
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


def sha(path):
    with path.open('rb') as handle:
        return file_digest(handle,'sha256').hexdigest()


def fetch(inputs):
    manifest=json.loads((inputs/'SCIENCEQA_SOURCE.json').read_text())
    for name,record in manifest['files'].items():
        path=inputs/name
        if not path.exists():
            tmp=path.with_suffix('.download')
            request=Request(record['url'],headers={'User-Agent':'ouro-joint-research'})
            with urlopen(request,timeout=45) as source,tmp.open('wb') as target:
                shutil.copyfileobj(source,target)
            if sha(tmp)!=record['sha256']:
                raise ValueError(f'ScienceQA source SHA mismatch for {name}')
            tmp.replace(path)
        elif sha(path)!=record['sha256']:
            raise ValueError(f'ScienceQA cached source SHA mismatch for {name}')
        print(name,record['sha256'],flush=True)


if __name__=='__main__':
    fetch(Path('inputs'))
