"""Verify multipart R7 evidence, restore only absent or byte-identical files."""
import argparse,hashlib,tarfile
from r7_core import ROOT,read,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    index=read('artifacts/r7_RECORDS_INDEX.json');assert index['status']=='verified'
    assert sha('artifacts/r7_RECORDS_CONTENTS.json')==index['contents_sha256']
    archive=ROOT/index['archive_path'];assert archive.parent==ROOT/'artifacts'
    for part in index['parts']:assert sha(part['path'])==part['sha256'] and (ROOT/part['path']).stat().st_size==part['bytes']
    if not archive.exists():
        with archive.open('xb') as out:
            for part in index['parts']:
                with (ROOT/part['path']).open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):out.write(block)
    assert sha(index['archive_path'])==index['archive_sha256'] and archive.stat().st_size==index['archive_bytes']
    records=read('artifacts/r7_RECORDS_CONTENTS.json')['files']
    with tarfile.open(archive,'r:xz') as tar:
        assert tar.getnames()==[r['path'] for r in records]
        for member,entry in zip(tar.getmembers(),records):
            target=ROOT/entry['path'];assert target.resolve().is_relative_to(ROOT.resolve()) and target.parent.is_dir()
            assert member.isfile() and not member.issym() and not member.islnk()
            raw=tar.extractfile(member).read();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
            if target.exists():assert sha(entry['path'])==entry['sha256'],'Existing different file will not be overwritten: '+entry['path']
            elif not a.verify_only:
                with target.open('xb') as out:out.write(raw)
    print('R7_RECORDS_VERIFIED',len(records),flush=True)


if __name__=='__main__':main()
