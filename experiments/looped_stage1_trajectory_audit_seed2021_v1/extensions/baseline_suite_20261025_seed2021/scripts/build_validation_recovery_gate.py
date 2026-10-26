"""Require exact replay, postprocessing and a captured zero child exit before resume."""
import hashlib
import json
import subprocess
import time
from common import ROOT,BASE,PROTOCOL_SHA,save

probe_exit=json.loads((ROOT/'logs/recovery_probe_exit_attempt2.json').read_text())
assert probe_exit['exit_code']==0 and not probe_exit['timeout']
probe=json.loads((ROOT/'results/recovery_boundary_replay.json').read_text())
assert probe['passed'] and probe['next_case_completed'] and probe['next_case_metadata_numeric_and_append_checked']
assert probe['completed_unix']>=probe_exit['started_unix']
assert len(probe['existing_replays'])==3 and all(r['token_ids_exact'] and r['text_exact'] and r['prediction_exact'] for r in probe['existing_replays'])
check=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
    "Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python' -and ($_.CommandLine -like '*baseline_suite_20261025_seed2021*' -or $_.CommandLine -like '*loopbase_env_20261025*') -and $_.CommandLine -match 'generate_math[.]py|actual_halting[.]py|run_suite[.]py|recovery_boundary_replay[.]py'}|Select-Object ProcessId,CommandLine|ConvertTo-Json"],capture_output=True,text=True,check=True)
assert not check.stdout.strip(),'A suite GPU owner/controller is still live'
files=[ROOT/'scripts/common.py',ROOT/'scripts/methods.py',BASE/'scripts/qwen_recipe.py',ROOT/'scripts/generate_math.py']
fingerprint=hashlib.sha256(''.join(p.name+hashlib.sha256(p.read_bytes()).hexdigest() for p in files).encode()).hexdigest()
prefixes={}
for name in ['Ouro-1.4B','Ouro-2.6B']:
    audit=json.loads((ROOT/f'results/math_prefix_audit_{name}.json').read_text())
    assert audit['passed'] and audit['ignored_unterminated_trailing_bytes']==0 and audit['protocol_sha256']==PROTOCOL_SHA
    assert audit['generation_implementation_sha256']==fingerprint
    path=ROOT/f'trajectories/math_{name}_validation.jsonl';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==audit['audited_prefix_sha256'] and not (ROOT/f'trajectories/math_{name}_test.jsonl').exists()
    assert json.loads((ROOT/f'results/generation_parity_{name}.json').read_text())['passed']
    prefixes[name]=dict(records=audit['audited_records'],sha256=digest,complete_validation=audit['complete_validation'])
assert prefixes['Ouro-1.4B']['sha256']==probe['formal_prefix_sha256']
budget=json.loads((ROOT/'logs/gpu_budget.json').read_text())
assert any(r.get('pid')==30772 and r['state']=='native_abort_0xc0000409' for r in budget['runs'])
assert any(r.get('pid')==probe['probe_pid'] and r['state']=='finished' for r in budget['runs'])
assert abs(budget['used_seconds']-budget['stage1_used_seconds']-sum(r['seconds'] for r in budget['runs']))<1e-4
assert budget['used_seconds']<50*3600
save('results/native_abort_validation_recovery_gate.json',dict(passed=True,checked_unix=time.time(),
     protocol_sha256=PROTOCOL_SHA,generation_implementation_sha256=fingerprint,validation_prefixes=prefixes,
     failed_worker_accounted=True,boundary_replays_exact=True,next_case_end_to_end_checked=True,
     diagnostic_native_exit_code=probe_exit['exit_code'],cumulative_gpu_hours=budget['used_seconds']/3600,
     scope='Recovery gate only; remaining validation/test/real stopping and final delivery remain required'))
print(json.dumps(dict(passed=True,preserved=prefixes,cumulative_gpu_hours=budget['used_seconds']/3600)),flush=True)
