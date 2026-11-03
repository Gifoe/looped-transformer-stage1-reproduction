"""Verify public archives and restore exact records/weight bundles using stdlib.

Existing differing files and paths outside V2 are rejected; no files are deleted.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def install(name,source):
    relative=Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    destination=(ROOT/relative).resolve()
    assert destination.is_relative_to(ROOT.resolve()) and destination!=ROOT.resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        assert destination.is_file() and digest(destination)==digest(source),('Preserve differing existing file',name)
    else:shutil.copyfile(source,destination)


def main():
    index=json.loads((ROOT/'ARTIFACT_INDEX.json').read_text(encoding='utf8'))
    (ROOT/'tmp').mkdir(exist_ok=True)
    assert (ROOT/'tmp').resolve().is_relative_to(ROOT.resolve())
    with tempfile.TemporaryDirectory(prefix='artifact_restore_',dir=ROOT/'tmp') as tmp:
        work=Path(tmp).resolve();assert work.is_relative_to(ROOT.resolve());restored=0
        record=index['records_archive'];archive=ROOT/record['path']
        assert digest(archive)==record['sha256']
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():continue
                assert ((info.external_attr>>16)&0o170000)!=0o120000,'No archive symlinks'
                target=work/'record';target.write_bytes(z.read(info))
                install(info.filename,target);restored+=1
        weights=index['weight_archive'];assembled=work/'exact_bundles.tar.xz'
        with assembled.open('wb') as out:
            for part in weights['parts']:
                path=ROOT/part['path'];assert path.stat().st_size==part['bytes'] and digest(path)==part['sha256']
                with path.open('rb') as f:shutil.copyfileobj(f,out)
        assert digest(assembled)==weights['sha256']
        with tarfile.open(assembled,'r:xz') as t:
            for info in t:
                if info.isdir():continue
                assert info.isfile() and not info.issym() and not info.islnk(),'No archive links/devices'
                target=work/'weight'
                with t.extractfile(info) as f,target.open('wb') as out:shutil.copyfileobj(f,out)
                install(info.name,target);restored+=1
    print(json.dumps(dict(status='restored',files=restored,exact_original_bytes_verified=True)))


if __name__=='__main__':main()
