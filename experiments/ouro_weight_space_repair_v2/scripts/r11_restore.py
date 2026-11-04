"""Safe byte-verified restore of the published R11 numerical/source archive."""
import argparse,hashlib,io,json,tarfile
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]


def relative(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive path: '+name)
    target=ROOT.joinpath(*p.parts).resolve()
    if not target.is_relative_to(ROOT.resolve()):raise ValueError('Archive path escapes existing experiment directory')
    return target


def main(verify_only):
    idx=json.loads((ROOT/'artifacts/r11_RECORDS_INDEX.json').read_text(encoding='utf8'));contents=ROOT/'artifacts/r11_RECORDS_CONTENTS.json'
    assert hashlib.sha256(contents.read_bytes()).hexdigest()==idx['contents_sha256']
    inherited=ROOT/'artifacts/r11_INHERITED_EVIDENCE.json'
    assert hashlib.sha256(inherited.read_bytes()).hexdigest()==idx['inherited_evidence_sha256']
    manifest=json.loads(contents.read_text(encoding='utf8'));expected={r['path']:r for r in manifest['files']};assert len(expected)==manifest['n']
    assert expected['artifacts/r11_MODEL_SNAPSHOT_METADATA.json']['sha256']==idx['model_snapshot_metadata_sha256']
    packed=[]
    for part in idx['parts']:
        raw=relative(part['path']).read_bytes();assert len(raw)==part['bytes'] and hashlib.sha256(raw).hexdigest()==part['sha256'];packed.append(raw)
    raw=b''.join(packed);assert len(raw)==idx['archive_bytes'] and hashlib.sha256(raw).hexdigest()==idx['archive_sha256']
    restored=0;present=0
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:xz') as tar:
        members=tar.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
        validated=[];targets=set()
        # Validate the entire archive and all existing files before any writes.
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk();entry=expected[member.name];target=relative(member.name)
            key=str(target).casefold();assert key not in targets,'Multiple archive names resolve to the same target';targets.add(key)
            value=tar.extractfile(member).read();assert len(value)==entry['bytes'] and hashlib.sha256(value).hexdigest()==entry['sha256']
            if target.exists():
                assert target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256'],('Existing different file refused',member.name);present+=1
            validated.append((target,value))
        if not verify_only:
            for target,value in validated:
                if target.exists():continue
                assert target.parent.is_dir(),'Restore into the existing project structure; no new experiment folders'
                with target.open('xb') as stream:stream.write(value)
                restored+=1
    print(json.dumps(dict(status='verified',verify_only=verify_only,archive_files=len(expected),existing_identical=present,restored=restored)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args();main(a.verify_only)
