"""Preserve a terminal native-abort checkpoint and conservatively close its budget."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'logs/recovery_validation1p4_20261026'
target.mkdir(parents=True,exist_ok=True)
status=json.loads((ROOT/'STATUS.json').read_text())
active=json.loads((ROOT/'logs/active_gpu.json').read_text())
assert status['stage']=='failed' and '3221226505' in status['error']
assert active['pid']==30772 and active['tag']=='generation_Ouro-1.4B_validation'
probe=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
    "Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python' -and ($_.CommandLine -like '*scripts/generate_math.py*' -or $_.CommandLine -like '*scripts/actual_halting.py*' -or $_.CommandLine -like '*scripts/run_suite.py*')}|Select-Object ProcessId,CommandLine|ConvertTo-Json"],capture_output=True,text=True,check=True)
assert not probe.stdout.strip(),'A suite process exists; do not reconcile/relaunch'
audit=json.loads((ROOT/'results/math_prefix_audit_Ouro-1.4B.json').read_text())
raw=ROOT/'trajectories/math_Ouro-1.4B_validation.jsonl'
payload=raw.read_bytes()
assert audit['passed'] and audit['audited_records']==835 and audit['ignored_unterminated_trailing_bytes']==0
assert hashlib.sha256(payload).hexdigest()==audit['audited_prefix_sha256']
manifest={}
for name in ['STATUS.json','logs/active_gpu.json','logs/gpu_budget.json',
             'logs/generation_validation_Ouro-1.4B.log','logs/math_supervisor_stdout.log',
             'logs/math_supervisor_stderr.log','results/math_prefix_audit_Ouro-1.4B.json',
             'trajectories/math_Ouro-1.4B_validation.jsonl']:
    source=ROOT/name;destination=target/name.replace('/','__')
    data=source.read_bytes()
    if destination.exists():assert destination.read_bytes()==data,'Existing recovery snapshot differs; inspect'
    else:shutil.copyfile(source,destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest()==hashlib.sha256(data).hexdigest()
    manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
events=[json.loads(line) for line in (ROOT/'logs/math_supervisor_stdout.log').read_text().splitlines() if line.startswith('{')]
start=next(e for e in reversed(events) if e.get('stage')=='generation_validation_Ouro-1.4B' and e.get('state')=='running')
end=next(e for e in reversed(events) if e.get('stage')=='failed')
assert end['updated_unix']==status['updated_unix'] and end['pid']==22184
seconds=end['updated_unix']-min(start['updated_unix'],active['started_unix'])
assert 0<seconds<50*3600
budget=json.loads((ROOT/'logs/gpu_budget.json').read_text())
assert not any(r.get('pid')==30772 and r['tag']==active['tag'] for r in budget['runs']),'Already charged; inspect existing receipt'
budget['runs'].append(dict(tag=active['tag'],pid=30772,seconds=seconds,state='native_abort_0xc0000409',
                          accounting='Conservative complete supervisor stage interval, including process startup; original worker could not close Budget',
                          started_unix=start['updated_unix'],finished_unix=end['updated_unix']))
budget['used_seconds']+=seconds
budget['extension_used_seconds']=budget['used_seconds']-budget['stage1_used_seconds']
assert abs(budget['used_seconds']-budget['stage1_used_seconds']-sum(r['seconds'] for r in budget['runs']))<1e-4
assert budget['used_seconds']<50*3600
temp=ROOT/'logs/gpu_budget.recovery.tmp';temp.write_text(json.dumps(budget,indent=2));temp.replace(ROOT/'logs/gpu_budget.json')
active.update(state='failed',finished_unix=end['updated_unix'],native_abort_budget_reconciled=True)
(ROOT/'logs/active_gpu.json').write_text(json.dumps(active,indent=2))
result=dict(passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),preserved_files=manifest,
            valid_complete_records=835,raw_validation_unchanged=True,charged_failed_seconds=seconds,
            cumulative_gpu_hours=budget['used_seconds']/3600,root_cause='Unestablished; Windows WER identifies ucrtbase.dll, exception c0000409/parameter5',
            next_step='Boundary replay and fresh recovery gate before identical-code validation resume')
(target/'RECOVERY_SNAPSHOT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(passed=True,records=835,charged_failed_seconds=seconds,cumulative_gpu_hours=budget['used_seconds']/3600)),flush=True)
