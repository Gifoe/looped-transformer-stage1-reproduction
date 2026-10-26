"""Gate the second model's recovery without overwriting first-model evidence."""
import hashlib
import json
import subprocess
import time
from common import ROOT, PROTOCOL_SHA, save
from audit_math_prefix import generation_fingerprint

assert not (ROOT/'results/native_abort_validation2p6_gate.json').exists(), 'Historical gate already exists; preserve it and use fresh diagnostics for any new recovery'

exit_record = json.loads((ROOT/'logs/native26_probe_exit.json').read_text())
probe = json.loads((ROOT/'results/native26_recovery_boundary_replay.json').read_text())
assert exit_record['exit_code'] == 0 and not exit_record['timeout']
assert probe['passed'] and probe['model'] == 'Ouro-2.6B'
assert probe['next_case_completed'] and probe['next_case_metadata_numeric_and_append_checked']
assert probe['formal_jsonl_unchanged'] and probe['formal_prefix_records'] == 207
assert exit_record['started_unix'] <= probe['completed_unix'] <= exit_record['finished_unix']
assert len(probe['existing_replays']) == 3
assert all(r['token_ids_exact'] and r['text_exact'] and r['prediction_exact'] for r in probe['existing_replays'])
assert sum(r['all_depth_argmax_exact'] for r in probe['existing_replays']) == 2
owner = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
    "Get-CimInstance Win32_Process|Where-Object{$_.Name -eq 'python.exe' -and $_.CommandLine -match 'generate_math[.]py|actual_halting[.]py|run_suite[.]py|native26_boundary_replay[.]py|native26_probe_controller[.]py'}|Select-Object ProcessId,CommandLine|ConvertTo-Json"],
    capture_output=True, text=True, check=True)
assert not owner.stdout.strip(), 'A possible experiment owner exists'
fingerprint = generation_fingerprint()
prefixes = {}
for name, count in [('Ouro-1.4B', 900), ('Ouro-2.6B', 207)]:
    audit = json.loads((ROOT/f'results/math_prefix_audit_{name}.json').read_text())
    assert audit['passed'] and audit['audited_records'] == count
    assert audit['ignored_unterminated_trailing_bytes'] == 0
    assert audit['protocol_sha256'] == PROTOCOL_SHA and audit['generation_implementation_sha256'] == fingerprint
    digest = hashlib.sha256((ROOT/f'trajectories/math_{name}_validation.jsonl').read_bytes()).hexdigest()
    assert digest == audit['audited_prefix_sha256']
    assert not (ROOT/f'trajectories/math_{name}_test.jsonl').exists()
    assert json.loads((ROOT/f'results/generation_parity_{name}.json').read_text())['passed']
    prefixes[name] = dict(records=count, sha256=digest, complete_validation=(count == 900))
assert prefixes['Ouro-2.6B']['sha256'] == probe['formal_prefix_sha256']
assert json.loads((ROOT/'results/recovery_lineage_check.json').read_text())['passed']
snapshot = json.loads((ROOT/'logs/recovery_validation2p6_20261026/RECOVERY_SNAPSHOT.json').read_text())
assert snapshot['passed'] and snapshot['valid_complete_records'] == 207
assert snapshot['preserved_files']['trajectories/math_Ouro-2.6B_validation.jsonl']['sha256'] == probe['formal_prefix_sha256']
budget = json.loads((ROOT/'logs/gpu_budget.json').read_text())
assert any(r.get('pid') == 25608 and r['state'] == 'native_abort_0xc0000005' for r in budget['runs'])
assert any(r.get('pid') == probe['probe_pid'] and r['state'] == 'finished' for r in budget['runs'])
assert abs(budget['used_seconds']-budget['stage1_used_seconds']-sum(r['seconds'] for r in budget['runs'])) < 1e-4
assert budget['used_seconds'] < 50*3600
result = dict(passed=True, checked_unix=time.time(), recovery_model='Ouro-2.6B',
    protocol_sha256=PROTOCOL_SHA, generation_implementation_sha256=fingerprint,
    validation_prefixes=prefixes, failed_worker_accounted=True, boundary_replays_exact=True,
    next_case_end_to_end_checked=True, diagnostic_native_exit_code=0,
    cumulative_gpu_hours=budget['used_seconds']/3600,
    scope='Recovery gate only. Root cause remains unestablished; remaining validation/test/actual stopping and full delivery are required.')
save('results/native_abort_validation2p6_gate.json', result)
print(json.dumps(result), flush=True)
