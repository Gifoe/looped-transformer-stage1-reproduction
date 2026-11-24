"""Download immutable public reading datasets with content verification."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path

import pyarrow.parquet as pq
import requests


SOURCES={
    'boolq': ('google/boolq','8ab2d3e','data',9427,3270,'4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623'),
    'social_iqa': ('allenai/social_i_qa','6db3dcea20d5430c3dd74bf85fe9ca87268ea91b','data',33410,1954,'4e3ec841d22f62a02a7a1c024bde076d735c5adc580a3455d70ae89d0144b2d0'),
    'race': ('ehovy/race','05d5c34d00dcd337e897d86ed88ad4f9d787857d','all',87866,4887,'d9986432dc7b5ced3d3e9372573eafdcfa599920bb5e65bbbdc245df3abfd09e'),
}


def checksum(path):
    with path.open('rb') as fio:
        return file_digest(fio,'sha256').hexdigest()


def download(out,source,split,mirror):
    repo,revision,config,train,dev,locked=SOURCES[source]
    expected=train if split=='train' else dev
    suffix=f'/datasets/{repo}/resolve/{revision}/{config}/{split}-00000-of-00001.parquet'
    logical='https://huggingface.co'+suffix
    name=out/f'{source}_{split}_raw.parquet'
    tmp=name.with_suffix('.partial')
    if not name.exists():
        for _ in range(3):
            try:
                with requests.get(mirror+suffix,stream=True,timeout=(12,90)) as response:
                    response.raise_for_status()
                    with tmp.open('wb') as fh:
                        for block in response.iter_content(1024*1024):
                            fh.write(block)
                break
            except requests.RequestException as error:
                print('retry',source,split,type(error).__name__,flush=True)
        else:
            raise RuntimeError(f'Failed to fetch {logical}')
        actual=checksum(tmp)
        if (split=='train' and actual!=locked) or pq.ParquetFile(tmp).metadata.num_rows!=expected:
            raise ValueError(f'Unexpected archived file hash or population for {source}/{split}')
        tmp.replace(name)
    actual=checksum(name)
    if (split=='train' and actual!=locked) or pq.ParquetFile(name).metadata.num_rows!=expected:
        raise ValueError(f'Changed data for {source}/{split}')
    result=dict(source=source,split=split,path=str(name),rows=expected,sha256=actual,repository=repo,
                revision=revision,url=logical)
    print(source,split,expected,actual,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--mirror',default='https://huggingface.co')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    tasks=[(a.out,source,split,a.mirror.rstrip('/')) for source in SOURCES for split in ('train','validation')]
    with ThreadPoolExecutor(max_workers=4) as threads:
        records=list(threads.map(lambda args:download(*args),tasks))
    report=a.out/'READING_INPUTS.json'
    report.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
