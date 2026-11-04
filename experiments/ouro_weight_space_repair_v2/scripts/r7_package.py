"""Portable numerical evidence archive. Flat existing directories, no secrets."""
import hashlib,io,tarfile
from r7_core import ROOT,read,save,sha


def main():
    delivery=read('results/r7_DELIVERABLES.json');assert delivery['status']=='rendered_from_complete_ARC'
    assert read('results/r7_TEST_ACCEPTANCE.json')['status']=='passed' and read('results/r7_PRESERVATION_AUDIT.json')['status']=='passed'
    if delivery['external_gate_open'] and not delivery.get('external_cancelled_by_user'):
        assert delivery['external_completed'],'Full external evaluation is required after passing ARC'
    if delivery.get('external_cancelled_by_user'):
        assert read('logs/r7_SCOPE_CHANGE_EXTERNAL_STOP.json')['status']=='external_evaluations_stopped_by_user'
        assert read('logs/r7_AFTER_ARC_STATUS.json')['status']=='external_cancelled_by_user'
    files=[]
    for folder in ['scripts','configs','results','logs']:
        for p in (ROOT/folder).iterdir():
            if not p.is_file() or not p.name.startswith('r7_') or p.suffix not in ['.py','.json','.jsonl','.csv','.md','.png','.pdf','.out','.err']:continue
            if p.name=='r7_COMPLETION_AUDIT.json':continue  # It verifies this archive, so cannot be inside it.
            # SSH observations and launch requests are transport records, not
            # experiment execution logs. Keep actual worker/coordinator outputs.
            if folder=='logs' and p.suffix=='.json' and not (p.name.endswith('_SUPERVISOR.json') or p.name in ['r7_gpu_budget.json','r7_PRESERVATION_BASELINE.json','r7_PIPELINE_STATUS.json','r7_AFTER_ARC_STATUS.json','r7_COORDINATOR_RESTART.json','r7_SCOPE_CHANGE_EXTERNAL_STOP.json']):continue
            if folder=='logs' and p.name in ['r7_after_arc_launcher.out','r7_after_arc_launcher.err']:continue
            if folder=='logs' and p.name.startswith('r7_arc_delivery_'):continue  # Active publication stdout must not self-change inside its archive.
            files.append(p)
    files.extend(ROOT.glob('r7_*.md'));files.append(ROOT/'data/DATA_SOURCE_NOTICE.txt')
    files.extend(ROOT/'scripts'/name for name in ['runtime.py','ouro_cached_scoring.py','slcd_r6_scores.py','weight_edits.py'])
    files=sorted(set(files));records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p.relative_to(ROOT))) for p in files]
    assert not any('/transport/' in r['path'] or r['path'].endswith('.pt') or '/models/' in r['path'] for r in records)
    save('artifacts/r7_RECORDS_CONTENTS.json',dict(status='complete',files=records,n=len(records),primary=delivery['primary'],arc_correct=delivery['arc_correct'],
        complete_experiment=delivery['complete_experiment'],external_cancelled_by_user=delivery.get('external_cancelled_by_user',False),
        exclusions=['SSH/transport/authentication','Model weights/environment','Raw historical hidden states','HellaSwag text and reversible target IDs','Other experiment archives'],
        data_attribution='Inherited ARC/demo text retains CC BY-SA4.0 attribution in data/DATA_SOURCE_NOTICE.txt'))
    archive=ROOT/'artifacts/r7_records.tar.xz'
    with tarfile.open(archive,'w:xz',preset=6) as tar:
        for p,entry in zip(files,records):
            raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['sha256']
            info=tarfile.TarInfo(entry['path']);info.size=len(raw);info.mtime=0;info.mode=0o644;info.uid=0;info.gid=0;info.uname='';info.gname=''
            tar.addfile(info,io.BytesIO(raw))
    with tarfile.open(archive,'r:xz') as tar:
        assert tar.getnames()==[r['path'] for r in records]
        for entry in records:assert hashlib.sha256(tar.extractfile(entry['path']).read()).hexdigest()==entry['sha256']
    parts=[]
    with archive.open('rb') as stream:
        while True:
            raw=stream.read(4*1024*1024)
            if not raw:break
            name=f'artifacts/r7_records.part{len(parts)+1:02d}.xz';(ROOT/name).write_bytes(raw)
            parts.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    save('artifacts/r7_RECORDS_INDEX.json',dict(status='verified',archive_path='artifacts/r7_records.tar.xz',archive_bytes=archive.stat().st_size,
        archive_sha256=sha('artifacts/r7_records.tar.xz'),contents_sha256=sha('artifacts/r7_RECORDS_CONTENTS.json'),parts=parts,
        restoration='Concatenate listed part bytes in order; validate archive SHA256; tar -xJf r7_records.tar.xz in the existing V2 experiment directory'))
    print('R7_PACKAGE_VERIFIED',len(records),archive.stat().st_size,flush=True)


if __name__=='__main__':main()
