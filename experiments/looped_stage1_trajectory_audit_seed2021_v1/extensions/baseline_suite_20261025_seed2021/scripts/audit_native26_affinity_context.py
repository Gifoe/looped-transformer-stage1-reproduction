"""Independently verify a diagnostic archive, then audit all216 isolated rows."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX_SHA = 'b2bb4dd03966cebaba23d32f7aac6981a5c58f16900841318e44da6e98d8efd9'
FROZEN = 'dbe4af336c619e5437ea31b9df91276ab125986b1a8d18d99bf85c0465eeebcf'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    archive_path = ROOT/'transport/native26_affinity_evidence.zip'
    metadata_path = ROOT/'transport/native26_affinity_evidence_metadata.json'
    if not archive_path.exists():
        archive_path = ROOT/'logs/native26_affinity_context/evidence_snapshot.zip'
        metadata_path = ROOT/'logs/native26_affinity_context/evidence_snapshot_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    assert archive_path.stat().st_size == metadata['bytes']
    assert sha(archive_path.read_bytes()) == metadata['sha256']
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('EVIDENCE_MANIFEST.json'))
        assert len(manifest) == metadata['files']
        assert set(archive.namelist()) == set(manifest) | {'EVIDENCE_MANIFEST.json'}
        payloads = {name: archive.read(name) for name in manifest}
    for name, payload in payloads.items():
        assert len(payload) == manifest[name]['bytes'] and sha(payload) == manifest[name]['sha256'], name
        assert '..' not in Path(name).parts and not Path(name).is_absolute()
    retained = ROOT/'logs/native26_affinity_context'
    retained.mkdir(parents=True, exist_ok=True)
    if archive_path.resolve() != (retained/'evidence_snapshot.zip').resolve():
        shutil.copyfile(archive_path, retained/'evidence_snapshot.zip')
        shutil.copyfile(metadata_path, retained/'evidence_snapshot_metadata.json')
    def get(name):
        return json.loads(payloads[name])
    directory = 'logs/native26_affinity_context/'
    results = get(directory+'RESULTS.json'); done = get(directory+'DONE.json')
    owner = get(directory+'OWNER.json'); affinity = get(directory+'AFFINITY.json')
    actual = get(directory+'ecore1_child_affinity.json')
    inputs = get(directory+'ecore1_input_manifest.json')
    trial, = results['trials']
    assert get('logs/native26_affinity_context_exit.json')['exit_code'] == 0
    assert done['completed'] and done['formal_prefix_unchanged'] and done['trials'] == results['trials']
    assert trial['native_exit_code'] == 0 and not trial['timeout'] and trial['complete_records'] == 216
    assert trial['all_next_question_conditions_completed'] and trial['next_record_exact'] and trial['baseline_argmax_exact']
    assert trial['child_affinity_verified'] and trial['affinity_error'] is None
    assert actual['verified'] and trial['affinity_mask'] == actual['mask'] == affinity['mask'] == '0xffff0000'
    assert affinity['logical_cpu_ids'] == list(range(16, 32)) and affinity['controller_pid'] == owner['controller_pid']
    assert inputs['generation_implementation_sha256'] == FROZEN
    for name in ['scripts/common.py', 'scripts/methods.py', 'scripts/generate_math.py', 'configs/protocol.json', 'data/math_fewshot.jsonl']:
        assert sha((ROOT/name).read_bytes()) == inputs['copied_files'][name]['sha256'], name
    original = (ROOT/'data/math_validation.jsonl').read_bytes()
    expected_subset = b''.join(original.splitlines(keepends=True)[:24])
    assert payloads['diagnostic/data/math_validation.jsonl'] == expected_subset
    assert sha(original) == inputs['copied_files']['data/math_validation.jsonl']['original_sha256']
    formal = (ROOT/'trajectories/math_Ouro-2.6B_validation.jsonl').read_bytes()
    assert sha(formal) == PREFIX_SHA
    raw = payloads['diagnostic/trajectories/math_Ouro-2.6B_validation.jsonl']
    assert raw[:len(formal)] == formal and raw.endswith(b'\n')
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 216
    expected_next = json.loads((ROOT/'results/native26_recovery_next_case_probe.json').read_text())
    new = rows[207]
    assert new['record_key'] == expected_next['record_key']
    assert all(new[k] == expected_next[k] for k in ['token_ids', 'text', 'prediction', 'correct', 'numeric_correct', 'condition'])
    assert [[s['prediction'] for s in p['by_depth']] for p in new['token_trajectory']] == [[s['prediction'] for s in p['by_depth']] for p in expected_next['token_trajectory']]
    ledger = get('logs/gpu_budget.json')
    run, = [r for r in ledger['runs'] if r.get('pid') == actual['pid']]
    assert run['state'] == 'finished' and run['diagnostic_trial'] == 'ecore1'
    assert run['tag'] == 'generation_Ouro-2.6B_validation'
    assert abs(ledger['used_seconds']-ledger['stage1_used_seconds']-sum(r['seconds'] for r in ledger['runs'])) < 1e-4
    assert ledger['used_seconds'] < 50*3600
    # Build an explicitly isolated CPU audit root in the same Stage1 layout.
    # The original audit implementation is copied byte-for-byte and run normally.
    target = ROOT.parent/(ROOT.name+'_affinity_ecore1_cpu_audit')
    target.mkdir(exist_ok=True)
    for name in ['scripts/common.py', 'scripts/methods.py', 'scripts/generate_math.py', 'scripts/audit_math_prefix.py',
                 'configs/protocol.json', 'data/math_fewshot.jsonl', 'results/model_load_Ouro-2.6B.json']:
        destination = target/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    tokenizers = target/'sources/tokenizers/Ouro-2.6B'
    shutil.copytree(ROOT/'sources/tokenizers/Ouro-2.6B', tokenizers, dirs_exist_ok=True)
    for name in ['data/math_validation.jsonl', 'trajectories/math_Ouro-2.6B_validation.jsonl', 'DIAGNOSTIC_ONLY.json', 'STATUS.json']:
        destination = target/name; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads['diagnostic/'+name])
    log = ROOT/'logs/native26_affinity_independent_cpu_audit.log'
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.run([sys.executable, '-u', '-X', 'utf8', str(target/'scripts/audit_math_prefix.py'),
                                  '--model', 'Ouro-2.6B', '--require-complete'], cwd=target,
                                 env=dict(os.environ, PYTHONUTF8='1', CUDA_VISIBLE_DEVICES=''), stdout=stream, stderr=subprocess.STDOUT)
    assert process.returncode == 0, str(log)
    audit = json.loads((target/'results/math_prefix_audit_Ouro-2.6B.json').read_text())
    assert audit['passed'] and audit['audited_records'] == audit['expected_records'] == 216
    assert audit['complete_validation'] and audit['generation_implementation_sha256'] == FROZEN
    assert audit['audited_prefix_sha256'] == sha(raw)
    assert (ROOT/'trajectories/math_Ouro-2.6B_validation.jsonl').read_bytes() == formal
    receipt = dict(passed=True, checked_unix=time.time(), archive_metadata=metadata,
        diagnostic_records=216, new_diagnostic_records=9, natural_native_exit_code=0,
        affinity_mask=actual['mask'], diagnostic_worker_pid=actual['pid'],
        formal_prefix_records=207, formal_prefix_sha256=PREFIX_SHA,
        formal_outputs_unchanged=True, generation_implementation_sha256=FROZEN,
        protocol_sha256=sha((ROOT/'configs/protocol.json').read_bytes()),
        independent_audit=audit, cumulative_gpu_hours=ledger['used_seconds']/3600,
        scope='Independent216-row diagnostic verification on a declared24-question prefix; not formal100-question completion, not proof of the crash root cause or long-run stability; diagnostic rows are never merged into formal data')
    (ROOT/'results/native26_affinity_independent_audit.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
