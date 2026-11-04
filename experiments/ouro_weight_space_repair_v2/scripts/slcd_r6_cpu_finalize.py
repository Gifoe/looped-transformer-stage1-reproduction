"""Wait for existing external jobs, then audit/export/report/package; no GPU.

Does not start, stop or restart model workers. Preserves failed receipts and
stops on any missing/changed prerequisite. All paths stay in the existing root.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(relative):return json.loads((ROOT/relative).read_text(encoding='utf8'))


def save(relative,value):
    path=ROOT/relative;tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)


def preservation():
    before=read('logs/slcd_PRESERVATION_BASELINE.json');mismatches=[]
    for relative,expected in before.items():
        p=ROOT/relative
        if not p.is_file():mismatches.append(dict(path=relative,reason='missing'));continue
        actual=p.stat()
        if actual.st_size!=expected['bytes'] or actual.st_mtime_ns!=expected['mtime_ns']:
            mismatches.append(dict(path=relative,expected=expected,actual=dict(bytes=actual.st_size,mtime_ns=actual.st_mtime_ns)))
    final=read('configs/slcd_R6_FINAL_LOCK.json');model=ROOT/'models/Ouro-1.4B/model.safetensors'
    actual_sha=sha(model);expected_sha=final['model_identity']['safetensors_sha256']
    output=dict(status='passed' if not mismatches and actual_sha==expected_sha else 'failed',
                old_file_count=len(before),size_and_mtime_ns_unchanged=not mismatches,mismatches=mismatches,
                model_safetensors_sha256=actual_sha,expected_model_sha256=expected_sha,
                preservation_baseline_sha256=sha(ROOT/'logs/slcd_PRESERVATION_BASELINE.json'),
                source_sha256=sha(Path(__file__)),scope='Same previous 671 files, plus original model complete byte hash; new slcd namespace intentionally excluded from old baseline.')
    save('results/slcd_R6_PRESERVATION_AUDIT.json',output)
    assert output['status']=='passed',json.dumps(output)


def run(script,*args):
    print('CPU_STAGE',script,flush=True)
    subprocess.run([sys.executable,'-B','scripts/'+script,*args],cwd=ROOT,check=True,env=os.environ.copy())


def main():
    assert ROOT.is_dir() and os.name=='posix'
    started=time.time();deadline=started+16000
    for key,path in [('PYTHONPYCACHEPREFIX','cache/pycache'),('MPLCONFIGDIR','cache/matplotlib'),('HF_HOME','cache/hf'),('TMPDIR','tmp'),('XDG_CACHE_HOME','cache/xdg')]:
        assert (ROOT/path).is_dir();os.environ[key]=str(ROOT/path)
    while not (ROOT/'results/slcd_R6_EXTERNAL_CHAIN_COMPLETE.json').exists():
        assert time.time()<deadline,'CPU finalizer wait deadline; preserve existing GPU attempt'
        for task in ['hellaswag','openbookqa']:
            p=ROOT/f'logs/slcd_r6_{task}_joint_SUPERVISOR.json'
            if p.exists():
                status=json.loads(p.read_text())
                assert status['status'] in ['starting','running','complete'],'Existing external worker failed: '+task
        time.sleep(20)
    chain=read('results/slcd_R6_EXTERNAL_CHAIN_COMPLETE.json')
    assert all(chain['tasks'][task]['status']=='complete' for task in ['hellaswag','openbookqa'])
    run('slcd_r6_external_audit.py')
    run('slcd_r6_public_scores.py','export')
    run('slcd_r6_public_scores.py','verify')
    preservation()
    run('slcd_r6_make_deliverables.py')
    run('slcd_r6_package_records.py')
    run('slcd_r6_restore_records.py','--verify-only')
    save('results/slcd_R6_CPU_FINAL_COMPLETE.json',dict(status='complete',no_model_forwards=True,
        elapsed_wall_seconds=time.time()-started,waiting_included=True,
        source_sha256=sha(Path(__file__)),release_index_sha256=sha(ROOT/'artifacts/slcd_R6_RELEASE_INDEX.json'),
        external_acceptance_sha256=sha(ROOT/'results/slcd_R6_EXTERNAL_ACCEPTANCE.json'),
        preservation_sha256=sha(ROOT/'results/slcd_R6_PRESERVATION_AUDIT.json'),
        report_sha256=sha(ROOT/'results/slcd_REPORT_ZH.md'),publication_status='Pending GitHub upload and remote readback by parent agent'))
    print('R6_CPU_FINAL_ARTIFACTS_VERIFIED',flush=True)


if __name__=='__main__':main()
