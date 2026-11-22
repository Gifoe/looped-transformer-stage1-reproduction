"""Fetch and canonicalize version-pinned OpenBookQA without exposing fact annotations."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

from bridge import dump

REVISION='388097ea7776314e93a529163e0fea805b8a6454'
URL=f'https://huggingface.co/datasets/allenai/openbookqa/resolve/{REVISION}/main/test-00000-of-00001.parquet'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main(raw,out):
    raw.parent.mkdir(parents=True,exist_ok=True)
    if not raw.exists():
        with urllib.request.urlopen(URL,timeout=35) as reply:
            temp=raw.with_suffix('.tmp')
            temp.write_bytes(reply.read())
            temp.replace(raw)
    tab=pq.read_table(raw).to_pylist()
    assert len(tab)==len({x['id'] for x in tab})==500
    rows=[dict(question=r['question_stem'],id=r['id'],choices=r['choices'],answerKey=r['answerKey']) for r in tab]
    assert all(r['answerKey'] in r['choices']['label'] for r in rows)
    pq.write_table(pa.Table.from_pylist(rows),out)
    dump(out.with_suffix('.json'),dict(dataset='allenai/openbookqa',split='main/test',revision=REVISION,
              n=500,raw_sha256=sha(raw),canonical_sha256=sha(out),columns=list(rows[0])))
    print('OpenBookQA test metadata verified',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();main(a.raw,a.out)
