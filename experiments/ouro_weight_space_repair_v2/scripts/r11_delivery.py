"""Unattended post-test CPU delivery, never inference or model reselection."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from r11_core import ROOT,read,save,sha


def status(value,**fields):save('logs/r11_DELIVERY_STATUS.json',dict(status=value,pid=os.getpid(),updated_unix=time.time(),**fields))


def main():
    own_source_at_launch=sha('scripts/r11_delivery.py')
    status('waiting_complete_ARC',observed_worker_pid=839650)
    while True:
        s=read('logs/r11_CHAIN_STATUS.json')
        if s['status'] in ['failed_requires_inspection','No-Go']:raise RuntimeError(('Inference chain did not complete',s))
        if s['status']=='complete_ARC':
            p=Path('/proc')/str(s['coordinator_pid'])/'cmdline'
            if not p.exists() or not p.read_bytes():break
        # A running record alone is not a liveness claim. Record actual handle
        # state, without restarting a worker on an observation/settlement gap.
        p=ROOT/'logs/r11_arc_test_SUPERVISOR.json'
        if p.exists():
            v=read(str(p.relative_to(ROOT)));q=Path('/proc')/str(v['worker_pid'])/'cmdline'
            status('waiting_complete_ARC',worker_pid=v['worker_pid'],actual_process_present=q.exists() and bool(q.read_bytes()),supervisor_status=v['status'])
        time.sleep(20)
    assert sha('scripts/r11_delivery.py')==own_source_at_launch,'CPU delivery source changed after launch; inspect and explicitly restart'
    for script,phase in [('r11_accept.py','experimental_acceptance'),('r11_make_deliverables.py','report_figures_cases'),('r11_package.py','lossless_archive'),('r11_restore.py','restore_byte_verification')]:
        status('running_'+phase)
        command=[str(ROOT/'env/bin/python'),'-B','scripts/'+script]
        if script=='r11_restore.py':command.append('--verify-only')
        subprocess.run(command,cwd=ROOT,check=True)
    index=read('artifacts/r11_RECORDS_INDEX.json');contents=read('artifacts/r11_RECORDS_CONTENTS.json');assert index['status']=='verified_complete_frozen_experiment_archive'
    required=['r11_README.md','r11_SELF_ITERATION_LOG.md','results/r11_REPORT_ZH.md','configs/r11_FINAL_METHOD_LOCK.json',
        'results/r11_arc_full_test_results.csv','results/r11_arc_paired_statistics.csv','results/r11_paired_transitions.csv','results/r11_compute_costs.csv',
        'results/r11_failure_cases.jsonl','results/r11_rescue_retention.csv','results/r11_DEPLOYMENT_PROFILE.json','results/r11_EXPERIMENT_ACCEPTANCE.json',
        'scripts/r11_replay.py','scripts/r11_prepare_model.py','scripts/r11_restore.py','scripts/r11_worker.py','artifacts/r11_MODEL_SNAPSHOT_METADATA.json','results/r11_arc_results.png','results/r11_arc_results.pdf',
        'results/r11_paired_comparisons.png','results/r11_paired_comparisons.pdf','results/r11_repair_outcomes.png','results/r11_repair_outcomes.pdf']
    assert set(required)<={r['path'] for r in contents['files']} and all((ROOT/p).is_file() and (ROOT/p).stat().st_size>0 for p in required)
    assert all(sha(r['path'])==r['sha256'] for r in contents['files'])
    assert sha(index['archive_path'])==index['archive_sha256'] and sha('artifacts/r11_RECORDS_CONTENTS.json')==index['contents_sha256']
    assert sha('artifacts/r11_INHERITED_EVIDENCE.json')==index['inherited_evidence_sha256'] and sha('artifacts/r11_MODEL_SNAPSHOT_METADATA.json')==index['model_snapshot_metadata_sha256']
    whole=hashlib.sha256()
    for p in index['parts']:
        value=(ROOT/p['path']).read_bytes();assert len(value)==p['bytes'] and hashlib.sha256(value).hexdigest()==p['sha256'];whole.update(value)
    assert whole.hexdigest()==index['archive_sha256']
    save('results/r11_RELEASE_ACCEPTANCE.json',dict(status='verified_experiments_named_artifacts_and_archive',goal_success_declared=False,files=contents['n'],archive_bytes=index['archive_bytes'],
        archive_index_sha256=sha('artifacts/r11_RECORDS_INDEX.json'),experimental_acceptance_sha256=sha('results/r11_EXPERIMENT_ACCEPTANCE.json'),
        audit_source_sha256=own_source_at_launch,remaining_checks=['Human visual inspection of final plots','GitHub publication lease and readback']))
    status('ready_for_visual_review_and_GitHub',n=1172,primary=read('results/r11_ARC_COMPLETE.json')['primary'],correct=read('results/r11_ARC_COMPLETE.json')['correct'],
        archive_bytes=index['archive_bytes'],parts=len(index['parts']))


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed_requires_inspection',error=repr(error));raise
