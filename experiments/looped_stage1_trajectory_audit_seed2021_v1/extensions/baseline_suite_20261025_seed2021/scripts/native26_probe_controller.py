"""Capture native child exit for the single2.6 recovery probe."""
from pathlib import Path
import json
import os
import subprocess
import time

root = Path(__file__).resolve().parents[1]
exit_path = root/'logs/native26_probe_exit.json'
assert not exit_path.exists(), 'Inspect existing probe outcome instead of relaunching'
budget = json.loads((root/'logs/gpu_budget.json').read_text())
assert any(r.get('pid') == 25608 and r['state'] == 'native_abort_0xc0000005' for r in budget['runs'])
check = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
    "Get-CimInstance Win32_Process|Where-Object{$_.Name -eq 'python.exe' -and $_.CommandLine -match 'generate_math[.]py|actual_halting[.]py|run_suite[.]py|native26_boundary_replay[.]py'}|Select-Object ProcessId,CommandLine|ConvertTo-Json"],
    capture_output=True, text=True, check=True)
assert not check.stdout.strip(), 'A possible GPU owner exists'
started = time.time()
timed_out = False
with (root/'logs/native26_probe_stdout.log').open('w', encoding='utf-8') as out, (root/'logs/native26_probe_stderr.log').open('w', encoding='utf-8') as err:
    child = subprocess.Popen(['D:/loopbase_env_20261025/Scripts/python.exe', '-u', '-X', 'utf8',
                              '-X', 'faulthandler', str(root/'scripts/native26_boundary_replay.py')],
                             cwd=root, stdout=out, stderr=err,
                             env=dict(os.environ, PYTHONUTF8='1', PYTHONFAULTHANDLER='1'))
    (root/'logs/native26_probe_owner.json').write_text(json.dumps(dict(controller_pid=os.getpid(),
         child_redirector_pid=child.pid, started_unix=started), indent=2))
    try:
        code = child.wait(timeout=600)
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'], stdout=err, stderr=err)
        child.wait()
        code = 124
result = dict(controller_pid=os.getpid(), child_redirector_pid=child.pid,
              started_unix=started, finished_unix=time.time(), exit_code=code, timeout=timed_out)
exit_path.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
raise SystemExit(code)
