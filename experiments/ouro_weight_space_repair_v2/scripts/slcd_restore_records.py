"""Restore archived records into existing V2 parents, refusing changed files."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
index=json.loads((ROOT/'artifacts/slcd_RECORDS_CONTENTS.json').read_text())
with zipfile.ZipFile(ROOT/'artifacts/slcd_records.zip') as z:
    assert set(z.namelist())=={r['path'] for r in index['files']}
    for r in index['files']:
        p=(ROOT/r['path']).resolve();assert p.is_relative_to(ROOT.resolve()) and p.parent.is_dir()
        raw=z.read(r['path']);assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
        if p.exists():assert p.read_bytes()==raw,'Existing different file: '+str(p)
        else:p.write_bytes(raw)
print('SLCD_RECORDS_VERIFIED_AND_RESTORED',index['file_count'])
