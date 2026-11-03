"""CPU-only transparent publication package; preserve large raw states remotely."""
import csv
import hashlib
import importlib.metadata
import json
import lzma
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from runtime import ROOT,save


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def info(path):return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest(path))


def main():
    assert json.loads((ROOT/'ARTIFACT_AUDIT.json').read_text())['status']=='passed'
    assert (ROOT/'REPORT_ZH.md').exists()
    artifacts=ROOT/'artifacts';artifacts.mkdir(exist_ok=True)
    distributions={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True)
    save('logs/ENVIRONMENT.json',dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
        packages=dict(sorted(distributions.items(),key=lambda x:x[0].lower())),gpu=gpu.stdout.strip(),
        inference='BF16; batch1; SDPA; no cache; TF32disabled; no explicit all-ones attention mask'))
    (ROOT/'requirements_frozen.txt').write_text(''.join(f'{n}=={v}\n' for n,v in sorted(distributions.items(),key=lambda x:x[0].lower())),encoding='utf8')
    # Preserve upstream source notices without copying the complete original model.
    source=ROOT/'sources/pinned_model';source.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'models/Ouro-1.4B').iterdir():
        if p.is_file() and (p.suffix=='.py' or p.name in ['config.json','README.md','snapshot_metadata.json'] or p.name.startswith(('LICENSE','NOTICE'))):
            destination=source/p.name
            if destination.exists():assert digest(p)==digest(destination)
            else:shutil.copy2(p,destination)
    provenance=json.loads((ROOT/'edited_weights/statistics/provenance.json').read_text())
    remote=[]
    for r in provenance['statistics_files']:
        p=ROOT/'results/training_statistics'/r['file'];assert p.is_file()
        remote.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=r['sha256'],
            uploaded=False,contents='K3-K6 raw forward statistics and answer-position recurrent states'))
    save('REMOTE_STATE_INDEX.json',dict(remote_experiment_directory='/home/newuser1/experiments/ouro_weight_space_repair_v2',
        preserved_raw_files=len(remote),bytes=sum(r['bytes'] for r in remote),files=remote,
        reason_not_in_git='Large raw diagnostic tensors; exact tested candidate weights/directions and full records are published.',
        regeneration='supervise.py -> collect_statistics.py --n384 using locked training manifests'))

    # All completed numerical stdout/stderr/receipts and all request/scoring records.
    # Orchestrator state and its own still-open package logs are excluded explicitly.
    excluded_names={'gpu.lock','FINISHER_STATUS.json','FINISHER_FAILURE.json',
        'package_results_launcher.out','package_results_launcher.err','finisher.out','finisher.err'}
    record_files=[]
    for folder in ['data','sources','logs','results']:
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if p.suffix=='.pt' or p.name in excluded_names:continue
            if folder=='logs' and (p.name.endswith('_launch.json') or p.name in ['statistics_progress.json','pipeline_progress.json']):continue
            record_files.append(p)
    records=artifacts/'reproduction_records.zip';assert not records.exists()
    with zipfile.ZipFile(records,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9,allowZip64=True) as z:
        for p in record_files:
            zi=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2026,11,3,0,0,0))
            zi.compress_type=zipfile.ZIP_DEFLATED;zi.external_attr=0o100644<<16
            z.writestr(zi,p.read_bytes())
    assert zipfile.ZipFile(records).testzip() is None
    save('artifacts/RECORDS_CONTENTS.json',dict(file_count=len(record_files),files=[info(p) for p in record_files]))

    # Archive all exact BF16 candidate bundles; long-range compression avoids
    # storing almost-identical full targets repeatedly. Original3GB model excluded.
    cid=json.loads((ROOT/'selected_edit_manifest.json').read_text())['selected_candidate_id']
    directories=[ROOT/'edited_weights/selected',ROOT/'edited_weights/candidates'/cid]
    directories.extend(p for p in sorted((ROOT/'edited_weights/candidates').iterdir()) if p.name!=cid)
    weight_files=[p for d in directories for p in sorted(d.iterdir()) if p.is_file()]
    temporary=ROOT/'tmp/exact_weight_bundles.tar.xz';assert not temporary.exists()
    with lzma.open(temporary,'xb',format=lzma.FORMAT_XZ,
        filters=[dict(id=lzma.FILTER_LZMA2,preset=6,dict_size=64*1024*1024)]) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as t:
            for p in weight_files:
                member=t.gettarinfo(str(p),arcname=p.relative_to(ROOT).as_posix())
                member.uid=member.gid=0;member.uname=member.gname='';member.mtime=0;member.mode=0o644
                with p.open('rb') as f:t.addfile(member,f)
    archive_sha=digest(temporary);parts=[]
    with temporary.open('rb') as f:
        number=0
        while True:
            block=f.read(4*1024*1024)
            if not block:break
            p=artifacts/f'exact_weight_bundles.tar.xz.part{number:03d}'
            with p.open('xb') as out:out.write(block)
            parts.append(info(p));number+=1
    # Validate archive structure and payload bytes before publication.
    with tarfile.open(temporary,'r:xz') as t:
        expected={p.relative_to(ROOT).as_posix():p for p in weight_files}
        assert len(t.getmembers())==len(expected)
        for member in t.getmembers():
            assert member.isfile() and member.name in expected
            h=hashlib.sha256()
            with t.extractfile(member) as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==digest(expected[member.name])
    save('artifacts/WEIGHTS_CONTENTS.json',dict(file_count=len(weight_files),files=[info(p) for p in weight_files]))
    direct=[]
    direct.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.csv','.pdf','.json','.txt'])
    for folder in ['configs','astra','figures','third_party','edited_weights/statistics','sources/pinned_model']:
        direct.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    direct.extend(p for p in (ROOT/'scripts').glob('*.py') if p.name not in ['ssh_transport.py','causal_hooks.py'])
    direct.extend(d/'manifest.json' for d in directories)
    direct.append(ROOT/'edited_weights/NOTICE.txt')
    direct.append(ROOT/'data/DATA_SOURCE_NOTICE.txt')
    direct.extend(artifacts.glob('*'))
    direct=sorted(set(direct));assert all(p.is_file() for p in direct)
    save('ARTIFACT_INDEX.json',dict(format='ouro_v2_public_artifacts_v1',
        records_archive=dict(**info(records),file_count=len(record_files)),
        weight_archive=dict(format='xz compressed tar split into ordered4MiB parts',bytes=temporary.stat().st_size,
            sha256=archive_sha,file_count=len(weight_files),parts=parts),
        direct_files=[info(p) for p in direct if p.name!='ARTIFACT_INDEX.json'],
        remote_only_raw_state_index='REMOTE_STATE_INDEX.json',
        original_model_included=False,environment_and_caches_included=False,
        all14_exact_candidate_bundles_published=True,complete_scoring_logs_published=True,
        excluded_orchestrator_logs=sorted(excluded_names),original_licenses='data/DATA_SOURCE_NOTICE.txt; edited_weights/NOTICE.txt'))
    direct=sorted(set(direct+[ROOT/'ARTIFACT_INDEX.json']))
    publication=ROOT/'publication';publication.mkdir(exist_ok=True)
    save('publication/PUBLIC_FILE_INDEX.json',dict(repository_prefix='experiments/ouro_weight_space_repair_v2',
        files=[info(p) for p in direct],total_bytes=sum(p.stat().st_size for p in direct)))
    save('results/PACKAGE_COMPLETE.json',dict(status='complete',public_files=len(direct),
        public_bytes=sum(p.stat().st_size for p in direct),records_bytes=records.stat().st_size,
        weight_archive_bytes=temporary.stat().st_size,weight_parts=len(parts),remote_raw_state_files=len(remote),
        exact_weight_archives_verified=True,records_zip_crc_verified=True))
    print(json.dumps(json.loads((ROOT/'results/PACKAGE_COMPLETE.json').read_text())),flush=True)


if __name__=='__main__':main()
