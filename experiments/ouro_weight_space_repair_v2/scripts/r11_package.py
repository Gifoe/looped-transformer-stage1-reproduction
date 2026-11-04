"""Lossless completed R11 evidence, source and logs. No models or SSH material."""
import hashlib,io,json,tarfile
from pathlib import Path
from r11_core import ROOT,read,sha,save


def main():
    accepted=read('results/r11_EXPERIMENT_ACCEPTANCE.json');assert accepted['status']=='passed_original_server_experiment_integrity'
    assert read('results/r11_ARC_COMPLETE.json')['n']==1172 and read('logs/r11_CHAIN_STATUS.json')['status']=='complete_ARC'
    # A writer must have exited before its stdout can be archived safely.
    pid=read('logs/r11_CHAIN_STATUS.json')['coordinator_pid'];q=Path('/proc')/str(pid)/'cmdline'
    assert not q.exists() or not q.read_bytes(),'Wait for completed coordinator stdout to close; never snapshot a live writer'
    allowed_logs={'r11_gpu_budget.json','r11_CHAIN_STATUS.json','r11_CPU_STATUS.json','r11_PUBLIC_CANDIDATE_RECEIPT.json','r11_PUBLIC_FINAL_LOCK_RECEIPT.json','r11_TEST_OBSERVATION_EXPOSURE.json'}
    files=[]
    for directory in ['scripts','configs','results','logs']:
        for p in (ROOT/directory).iterdir():
            if not p.is_file() or not p.name.startswith('r11_') or p.suffix not in ['.py','.json','.jsonl','.csv','.md','.png','.pdf','.out','.err']:continue
            if p.name in ['r11_RELEASE_ACCEPTANCE.json','r11_PUBLIC_RECORDS_ACCEPTANCE.json']:continue
            if directory=='logs':
                if p.suffix=='.json' and not (p.name.endswith('_SUPERVISOR.json') or p.name in allowed_logs):continue
                if p.name.startswith('r11_delivery_') or '_launcher.' in p.name:continue
            files.append(p)
    files.extend(ROOT.glob('r11_*.md'));files.append(ROOT/'data/DATA_SOURCE_NOTICE.txt')
    files.append(ROOT/'artifacts/r11_MODEL_SNAPSHOT_METADATA.json')
    candidate=read('configs/r11_CANDIDATE_LOCK.json')
    files.extend(ROOT/'scripts'/n for n in set(candidate['extra_source_sha256'])|set(candidate['inference_definition_sha256']))
    files=sorted(set(files));records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p.relative_to(ROOT))) for p in files]
    assert all('/transport/' not in r['path'] and '/models/' not in r['path'] and not r['path'].endswith('.pt') for r in records)
    inherited=dict(status='unchanged_inherited_evidence',repository='Gifoe/looped-transformer-stage1-reproduction',branch='codex/ouro-r7-prefix-prior-20261104',
        R6=dict(index='artifacts/slcd_R6_RELEASE_INDEX.json',archive_sha256='692a333601798e3a2569b70f108efbf440e04792d6b9ff4e22159bfb5ab7d952',purpose='Original ARC data/25demo manifests, source authority and original baselines'),
        R7=dict(index='artifacts/r7_RECORDS_INDEX.json',index_sha256=sha('artifacts/r7_RECORDS_INDEX.json'),archive_sha256=read('artifacts/r7_RECORDS_INDEX.json')['archive_sha256'],commit='320af4fb73103166ccc0eda72af0b774e7a03de2',purpose='Complete original main/N0 scores and pinned input/source evidence'),
        development=dict(index='artifacts/r10_r11_DEVELOPMENT_INDEX.json',archive_sha256=read('artifacts/r10_r11_DEVELOPMENT_INDEX.json')['archive']['sha256'],purpose='All576 auxiliary forwards and immutable R11 development receipts'),
        older_ledgers='Preserved public logs/r7_gpu_budget.json, logs/r9_gpu_budget.json, logs/r10_gpu_budget.json. Exact bytes are bound by the definition lock.',
        model_preparation=dict(script='scripts/r11_prepare_model.py',snapshot_metadata='artifacts/r11_MODEL_SNAPSHOT_METADATA.json',snapshot_metadata_sha256=sha('artifacts/r11_MODEL_SNAPSHOT_METADATA.json'),
            note='Custom small metadata is released; weights remain excluded. --prepare verifies all pinned bytes before creating absent runtime metadata and refuses changes to existing different files or external model symlinks.'),
        restoration='Restore inherited R6 then R7 exact evidence before R11. Mismatched existing files are refused; no old evidence is silently overwritten.')
    save('artifacts/r11_INHERITED_EVIDENCE.json',inherited)
    save('artifacts/r11_RECORDS_CONTENTS.json',dict(status='complete_frozen_R11_experimental_records',n=len(records),files=records,primary=accepted['primary'],
        exclusions=['Weights/environment','SSH/transport/authentication','Raw hidden states or corpus-vocabulary logits','Active publication stdout and self-referencing final archive acceptance'],
        data_attribution='ARC text retains CC BY-SA4.0 attribution in data/DATA_SOURCE_NOTICE.txt'))
    archive=ROOT/'artifacts/r11_records.tar.xz';assert not archive.exists(),'Inspect prior package before explicit rebuild'
    with tarfile.open(archive,'w:xz',preset=6) as tar:
        for p,r in zip(files,records):
            raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'];info=tarfile.TarInfo(r['path']);info.size=len(raw);info.mtime=0;info.mode=0o644;info.uid=info.gid=0;info.uname=info.gname='';tar.addfile(info,io.BytesIO(raw))
    with tarfile.open(archive,'r:xz') as tar:
        assert tar.getnames()==[r['path'] for r in records]
        for r in records:assert hashlib.sha256(tar.extractfile(r['path']).read()).hexdigest()==r['sha256']
    parts=[]
    with archive.open('rb') as stream:
        while True:
            raw=stream.read(4*1024*1024)
            if not raw:break
            name=f'artifacts/r11_records.part{len(parts)+1:02d}.xz';(ROOT/name).write_bytes(raw);parts.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    save('artifacts/r11_RECORDS_INDEX.json',dict(status='verified_complete_frozen_experiment_archive',archive_path='artifacts/r11_records.tar.xz',archive_bytes=archive.stat().st_size,
        archive_sha256=sha('artifacts/r11_records.tar.xz'),contents_sha256=sha('artifacts/r11_RECORDS_CONTENTS.json'),inherited_evidence_sha256=sha('artifacts/r11_INHERITED_EVIDENCE.json'),
        model_snapshot_metadata_sha256=sha('artifacts/r11_MODEL_SNAPSHOT_METADATA.json'),
        experiment_acceptance_sha256=sha('results/r11_EXPERIMENT_ACCEPTANCE.json'),parts=parts,
        restoration='Restore inherited exact inputs, concatenate listed parts, validate SHA256, then python -B scripts/r11_restore.py --verify-only or omit --verify-only to restore absent records'))
    assert all(sha(r['path'])==r['sha256'] for r in records),'A current file changed during publication; inspect live writers'
    print('R11_PACKAGE_VERIFIED',len(records),archive.stat().st_size,flush=True)


if __name__=='__main__':main()
