"""Fresh recovery gate after a naturally exited nine-condition affinity trial."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
from common import ROOT, PROTOCOL_SHA, save
from audit_math_prefix import generation_fingerprint

destination = 'results/native26_affinity_recovery_gate.json'
assert not (ROOT/destination).exists(), 'Never overwrite a historical recovery gate'
independent = json.loads((ROOT/'results/native26_affinity_independent_audit.json').read_text())
assert independent['passed'] and independent['formal_outputs_unchanged']
assert independent['diagnostic_records'] == 216 and independent['new_diagnostic_records'] == 9
assert independent['natural_native_exit_code'] == 0 and independent['affinity_mask'] == '0xffff0000'
out = ROOT/'logs/native26_affinity_context'
trial, = json.loads((out/'RESULTS.json').read_text())['trials']
assert trial['native_exit_code'] == 0 and not trial['timeout'] and trial['child_affinity_verified']
assert trial['next_record_exact'] and trial['baseline_argmax_exact'] and trial['complete_records'] == 216
assert json.loads((out/'DONE.json').read_text())['completed']
assert json.loads((ROOT/'logs/native26_affinity_context_exit.json').read_text())['exit_code'] == 0
assert hashlib.sha256((out/'evidence_snapshot.zip').read_bytes()).hexdigest() == independent['archive_metadata']['sha256']
fingerprint = generation_fingerprint()
assert independent['generation_implementation_sha256'] == fingerprint and independent['protocol_sha256'] == PROTOCOL_SHA
owner = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
    "Get-CimInstance Win32_Process|Where-Object{$_.Name -eq 'python.exe' -and $_.CommandLine -match 'generate_math[.]py|actual_halting[.]py|run_suite[.]py|native26_boundary_replay[.]py'}|Select-Object ProcessId,CommandLine|ConvertTo-Json"],
    capture_output=True, text=True, check=True)
assert not owner.stdout.strip(), 'An experiment owner is live'
prefixes = {}
for name, count in [('Ouro-1.4B', 900), ('Ouro-2.6B', 207)]:
    audit = json.loads((ROOT/f'results/math_prefix_audit_{name}.json').read_text())
    assert audit['passed'] and audit['audited_records'] == count and audit['ignored_unterminated_trailing_bytes'] == 0
    assert audit['protocol_sha256'] == PROTOCOL_SHA and audit['generation_implementation_sha256'] == fingerprint
    digest = hashlib.sha256((ROOT/f'trajectories/math_{name}_validation.jsonl').read_bytes()).hexdigest()
    assert digest == audit['audited_prefix_sha256']
    assert not (ROOT/f'trajectories/math_{name}_test.jsonl').exists()
    assert json.loads((ROOT/f'results/generation_parity_{name}.json').read_text())['passed']
    prefixes[name] = dict(records=count, sha256=digest, complete_validation=(count == 900))
assert prefixes['Ouro-2.6B']['sha256'] == independent['formal_prefix_sha256']
assert json.loads((ROOT/'results/recovery_lineage_check.json').read_text())['passed']
budget = json.loads((ROOT/'logs/gpu_budget.json').read_text())
for pid in [25608, 35364]:
    row, = [r for r in budget['runs'] if r.get('pid') == pid]
    assert row['state'] == 'native_abort_0xc0000005' and row['seconds'] > 0
row, = [r for r in budget['runs'] if r.get('pid') == independent['diagnostic_worker_pid']]
assert row['state'] == 'finished' and row['diagnostic_trial'] == 'ecore1'
assert abs(budget['used_seconds']-budget['stage1_used_seconds']-sum(r['seconds'] for r in budget['runs'])) < 1e-4
assert budget['used_seconds'] < 50*3600
result = dict(passed=True, checked_unix=time.time(), recovery_model='Ouro-2.6B',
    protocol_sha256=PROTOCOL_SHA, generation_implementation_sha256=fingerprint,
    validation_prefixes=prefixes, required_process_affinity_mask='0xffff0000',
    independently_audited_diagnostic_records=216, diagnostic_native_exit_code=0,
    diagnostic_archive_sha256=independent['archive_metadata']['sha256'],
    failed_worker_pids_accounted=[25608, 35364], cumulative_gpu_hours=budget['used_seconds']/3600,
    supersedes_invalidated_gate='results/native_abort_validation2p6_gate.json',
    scope='Fresh controlled recovery after216-row diagnostic with nine new conditions and exact known next-baseline tokens; no claim of root cause or guaranteed long-run stability. Formal900 validation, held-out tests, actual stopping and complete delivery remain required.')
save(destination, result)
print(json.dumps(result), flush=True)
