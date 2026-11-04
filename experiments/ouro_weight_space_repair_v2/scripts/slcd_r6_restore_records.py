"""Verify/restore final multipart records inside the existing experiment root.

Never invokes tar extraction on untrusted paths or overwrites different files.
Requires the release index, contents index, and every ordered archive part.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def safe_path(relative):
    p=(ROOT/relative).resolve();assert p.is_relative_to(ROOT.resolve()),'Path outside experiment root'
    return p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    release=json.loads((ROOT/'artifacts/slcd_R6_RELEASE_INDEX.json').read_text(encoding='utf8'))
    assert release['status']=='verified_complete_release'
    contents_path=safe_path(release['contents']['path']);assert digest_file(contents_path)==release['contents']['sha256']
    contents=json.loads(contents_path.read_text(encoding='utf8'));assert contents['file_count']==release['file_count']
    archive=safe_path(release['archive']['path']);parts=release['parts']
    assert parts and len({p['path'] for p in parts})==len(parts)
    for part in parts:
        p=safe_path(part['path']);assert p.stat().st_size==part['bytes'] and digest_file(p)==part['sha256']
    if not archive.exists():
        temp=archive.with_suffix(archive.suffix+'.tmp');assert not temp.exists()
        try:
            with temp.open('xb') as dst:
                for part in parts:
                    with safe_path(part['path']).open('rb') as src:
                        for b in iter(lambda:src.read(1024*1024),b''):dst.write(b)
            assert temp.stat().st_size==release['archive']['bytes'] and digest_file(temp)==release['archive']['sha256']
            temp.replace(archive)
        except BaseException:
            if temp.exists():temp.unlink()
            raise
    assert archive.stat().st_size==release['archive']['bytes'] and digest_file(archive)==release['archive']['sha256']
    with tarfile.open(archive,'r:xz') as tar:
        members=tar.getmembers();records=contents['files']
        assert [m.name for m in members]==[r['path'] for r in records]
        for member,r in zip(members,records):
            assert member.isfile() and not member.issym() and not member.islnk()
            destination=safe_path(r['path']);assert destination.parent.is_dir(),'Existing parent required: '+str(destination.parent)
            stream=tar.extractfile(member);assert stream is not None;raw=stream.read()
            assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
            if destination.exists():assert destination.is_file() and destination.read_bytes()==raw,'Existing different file: '+r['path']
            elif not args.verify_only:destination.write_bytes(raw)
    print(json.dumps(dict(status='verified',mode='verify_only' if args.verify_only else 'restore_absent_only',records=len(contents['files']),archive_sha256=release['archive']['sha256'])))


if __name__=='__main__':main()
