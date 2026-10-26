"""Verify the archive seal and every extracted file without loading a model."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):sha.update(block)
    return sha.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--zip',type=Path);args=parser.parse_args()
    metadata=json.loads((args.root/'DELIVERY.json').read_text());manifest=json.loads((args.root/'ARTIFACT_MANIFEST.json').read_text())
    assert digest(args.root/'ARTIFACT_MANIFEST.json')==metadata['manifest_sha256']
    for name,entry in manifest.items():
        path=args.root/name;assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],name
    if args.zip:
        assert args.zip.stat().st_size==metadata['bytes'] and digest(args.zip)==metadata['sha256']
        with zipfile.ZipFile(args.zip) as archive:
            assert archive.testzip() is None
            for name,entry in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==entry['sha256'],name
    print(json.dumps(dict(passed=True,files=len(manifest),archive_checked=bool(args.zip))),flush=True)

if __name__=='__main__':main()
