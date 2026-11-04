"""Package final evidence without model weights, state tensors or HH text.

Requires completed ARC, both external tasks and independent acceptance. Uses
existing directories only. Raw inputs/caches remain on server; public derived
external scores preserve every numerical likelihood/ranking without text.
"""
from pathlib import Path
import hashlib
import io
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
CHUNK=4*1024*1024


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    (ROOT/path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def read(path):return json.loads((ROOT/path).read_text(encoding='utf8'))


def eligible(path):
    relative=path.relative_to(ROOT).as_posix();folder=path.parent.name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):return False
    if folder=='scripts':return path.name.startswith('slcd_') and path.suffix=='.py' or path.name in ['runtime.py','ouro_cached_scoring.py','weight_edits.py']
    if folder=='configs':return path.name.startswith('slcd_') and path.suffix=='.json'
    if folder=='results':
        if not path.name.startswith('slcd_') or path.suffix not in ['.json','.jsonl','.csv','.md','.pdf','.png']:return False
        return path.name not in ['slcd_hellaswag_scores.jsonl','slcd_r6_hellaswag_scores.jsonl','slcd_r6_external_prior_cache.jsonl','slcd_R6_CPU_FINAL_COMPLETE.json']
    if folder=='data':
        if path.name=='DATA_SOURCE_NOTICE.txt':return True
        if not path.name.startswith('slcd_'):return False
        return 'hellaswag' not in path.name or path.name=='slcd_hellaswag_raw_SOURCE.json'
    if folder=='astra':return path.name.startswith('slcd_') and path.suffix in ['.json','.md','.py']
    if folder=='logs':return path.name.startswith('slcd_') and not path.name.startswith('slcd_r6_cpu_finalize') and (path.suffix in ['.out','.err'] or path.name.endswith('_SUPERVISOR.json') or path.name in ['slcd_gpu_budget.json','slcd_PRESERVATION_BASELINE.json'])
    if folder=='artifacts':return path.name=='slcd_R6_STATE_CACHE_INDEX.json' or (path.name.startswith('slcd_r6_') and path.suffix in ['.npz','.jsonl'])
    return False


def main():
    assert read('results/slcd_R6_TEST_ACCEPTANCE.json')['status']=='passed'
    assert read('results/slcd_R6_EXTERNAL_ACCEPTANCE.json')['status']=='passed'
    assert read('results/slcd_R6_PUBLIC_NUMERIC_ACCEPTANCE.json')['status']=='passed'
    chain=read('results/slcd_R6_EXTERNAL_CHAIN_COMPLETE.json')
    assert all(chain['tasks'][name]['status']=='complete' for name in ['hellaswag','openbookqa'])
    assert read('results/slcd_R6_DELIVERABLES.json')['status']=='rendered_from_complete_results'
    preservation=read('results/slcd_R6_PRESERVATION_AUDIT.json');assert preservation['status']=='passed'
    states=[]
    for p in sorted((ROOT/'results').glob('slcd_state_*.pt')):
        states.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    save('artifacts/slcd_R6_STATE_CACHE_INDEX.json',dict(scope='Regenerable acceleration states retained on server, not distributed',n=len(states),files=states))
    files=[]
    for folder in ['scripts','configs','results','data','astra','logs','artifacts']:
        files.extend(p for p in (ROOT/folder).iterdir() if eligible(p))
    files.extend(ROOT/p for p in ['FINAL_METHOD_LOCK.json','SELF_ITERATION_LOG.md','slcd_README.md','slcd_FOLLOWUP_STATUS.md','slcd_R6_ARC_STAGE_REPORT.md'])
    files=sorted(set(files));records=[]
    for p in files:
        records.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    # Verify exclusion rules explicitly, rather than relying on archive globbing.
    names=[r['path'] for r in records]
    assert not any(x.endswith('.pt') or '/models/' in x or '/env/' in x or '/transport/' in x for x in names)
    assert not any(x in names for x in ['data/slcd_hellaswag.jsonl','data/slcd_hellaswag_raw.parquet','results/slcd_r6_hellaswag_scores.jsonl','results/slcd_r6_external_prior_cache.jsonl','results/slcd_hellaswag_scores.jsonl'])
    contents=dict(status='complete',primary='r6_prior_projection',arc_correct=736,file_count=len(records),files=records,
                  excluded=['Model weights/environment','Raw hidden-state tensors, indexed separately','HellaSwag text/parquet/rendered inputs/reversible token IDs and raw text-containing cache/JSONL','SSH/transport/authentication files','Old nested archives and publication working metadata'],
                  lossless_external_numeric_receipt_sha256=sha(ROOT/'results/slcd_R6_PUBLIC_NUMERIC_EXPORT.json'),
                  full_originals_location='Same experiment server results/data folders; not all originals are redistributed')
    save('artifacts/slcd_R6_RECORDS_CONTENTS.json',contents)
    archive=ROOT/'artifacts/slcd_R6_records.tar.xz'
    with tarfile.open(archive,'w:xz',preset=6) as tar:
        for p,r in zip(files,records):
            raw=p.read_bytes();assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
            info=tarfile.TarInfo(r['path']);info.size=len(raw);info.mtime=0;info.uid=0;info.gid=0;info.uname='';info.gname='';info.mode=0o644
            tar.addfile(info,io.BytesIO(raw))
    # Reopen and verify every archived record before declaring a release.
    with tarfile.open(archive,'r:xz') as tar:
        members=tar.getmembers();assert [m.name for m in members]==names
        for member,r in zip(members,records):
            f=tar.extractfile(member);assert f is not None;raw=f.read()
            assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
    parts=[]
    with archive.open('rb') as f:
        i=0
        while True:
            block=f.read(CHUNK)
            if not block:break
            p=ROOT/f'artifacts/slcd_R6_records.tar.xz.part{i:03d}';p.write_bytes(block)
            parts.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=len(block),sha256=sha(p)));i+=1
    index=dict(status='verified_complete_release',primary='r6_prior_projection',arc_correct=736,
               archive=dict(path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive)),
               chunk_bytes=CHUNK,parts=parts,contents=dict(path='artifacts/slcd_R6_RECORDS_CONTENTS.json',sha256=sha(ROOT/'artifacts/slcd_R6_RECORDS_CONTENTS.json')),
               file_count=len(records),uncompressed_bytes=sum(r['bytes'] for r in records),
               frozen_final_sha256=sha(ROOT/'configs/slcd_R6_FINAL_LOCK.json'),external_acceptance_sha256=sha(ROOT/'results/slcd_R6_EXTERNAL_ACCEPTANCE.json'),
               raw_text_distribution_note=contents['excluded'])
    save('artifacts/slcd_R6_RELEASE_INDEX.json',index)
    print(json.dumps({k:v for k,v in index.items() if k not in ['raw_text_distribution_note']},indent=2))


if __name__=='__main__':main()
