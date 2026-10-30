"""Reviewable research delivery; deliberately excludes operational transports/caches/weights."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
from linux_runtime import ROOT,save


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    names=['README.md','REPORT.md','CASE_STUDIES.md','CURRENT_SCOPE.md','CAUSAL_FOLLOWUP_PROTOCOL.md',
        'CAUSAL_VALIDATION_PROTOCOL.md','REPRODUCE.md','hypothesis_matrix.md','NEXT_STEPS.md','ARC_FIRST_REPORT.md',
        'STATUS.json','checkpoint_manifest.json','compute_ledger.json','secondary_margin_effects.json']
    paths=[ROOT/n for n in names]+list(ROOT.glob('*.csv'))
    scripts=['analyze_arc_first.py','analyze_existing_transitions.py','analyze_causal_pilot.py','audit_mechanisms.py',
        'analyze_feedback_validation.py','analyze_margin_effects.py','audit_arc_nl23.py','audit_published.py',
        'audit_transfer_records.py','assemble_current_evidence.py','build_bench_parquet.py','build_final_report.py',
        'causal_trace.py','check_causal_trace.py','compare_nl23_nl33w.py','fetch_models_manifest.py',
        'inventory_rebuild_inputs.py','pin_coread_sources.py','linux_runtime.py','linux_cpu_gate.py','linux_prep_own.py',
        'mcq_tools.py','plot_audit.py','plot_arc_first.py','plot_mechanisms.py','trace_dynamics.py',
        'run_arc_first.py','run_arc_controls.py','run_checkpoint_pilot.py','run_causal_pilot.py',
        'run_feedback_validation.py','run_study_supervised.py','linux_arc_pipeline.py',
        'linux_causal_pilot_pipeline.py','nl20_warm_training_smoke.py','nl20_warm_training_smoke_v2.py','package_delivery.py']
    paths += [ROOT/'scripts'/n for n in scripts]
    paths += list((ROOT/'configs').glob('*.json'))
    paths += [ROOT/'configs/requirements_reproduction.txt']
    paths += [p for p in (ROOT/'fresh').rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.png','.pdf','.svg']]
    paths += [p for p in (ROOT/'figures').glob('*') if p.suffix in ['.png','.pdf','.svg']]
    paths += list((ROOT/'sources/published').rglob('*.json'))
    paths += [ROOT/'sources'/n for n in ['PRIOR_ART.md','arc_challenge_test_pinned.parquet',
        'upstream_runtime_7e7fcc4.zip','upstream_full_7e7fcc4.zip','RAW_ARCHIVE_MANIFEST.json']]
    logs=['linux_environment_manifest.json','TINY_TESTS_EXIT.json','CAUSAL_TRACE_CHECK.json',
        'CAUSAL_PILOT_STATE.json','causal_original_shape_STATE.json','feedback_validation_STATE.json','ARC_PIPELINE_STATE.json',
        'RELEASED_GATE_PARAMETERS.json','PUBLISHED_ARC_NL23.json','OWN_REPLY_STATE.json','WARM_TRAINING_SMOKE.json',
        'WARM_TRAINING_SMOKE_V2.json','LOCAL_RAW_ARCHIVE_VERIFIED.json','LOCAL_STATE_TENSORS_VERIFIED.json']
    paths += [ROOT/'logs'/n for n in logs]
    paths=sorted(set(paths))
    assert all(p.is_file() for p in paths),[str(p) for p in paths if not p.is_file()]
    # These operational addresses/credentials must not leak through the public research subset.
    forbidden=[r'(?:https?://|[\x22\x27\s])(?:10|100)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        r'(?:X-Amz|X-Xet)-Signature\s*=',r'password\s*[:=]\s*[\x22\x27][^\x22\x27]{3,}[\x22\x27]']
    for p in paths:
        if p.suffix in ['.json','.md','.py','.txt']:
            value=p.read_text(encoding='utf8')
            assert not any(re.search(s,value,re.IGNORECASE) for s in forbidden),(p,'operational data in public subset')
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}
    save(ROOT/'ARTIFACT_FILES_SHA256.json',manifest)
    dest=ROOT/'artifacts/neural_pairing_audit_code_results_20261030.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths+[ROOT/'ARTIFACT_FILES_SHA256.json']:z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    raw=ROOT/'artifacts/neural_pairing_audit_raw_20261030.tar.gz'
    expected=json.loads((ROOT/'sources/RAW_ARCHIVE_MANIFEST.json').read_text())
    assert sha(raw)==expected['sha256']
    delivery=dict(status='complete',code_results=dict(file=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),files=len(paths)+1),
        raw_data=dict(file=raw.name,bytes=raw.stat().st_size,sha256=expected['sha256']),
        instructions='Extract raw-data tar into an empty experiment directory, then unzip code/results over the same directory so final analyses/plots take precedence. No model weights, credentials or operational transport included.')
    save(ROOT/'artifacts/DELIVERY_MANIFEST.json',delivery);print(json.dumps(delivery,indent=2))


if __name__=='__main__':main()
