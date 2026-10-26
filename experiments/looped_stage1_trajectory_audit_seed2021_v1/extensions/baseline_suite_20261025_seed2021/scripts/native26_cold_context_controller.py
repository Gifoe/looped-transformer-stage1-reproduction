"""Run byte-identical producers in isolated output roots for a cold-start diagnosis.

Only diagnostic copies receive new predictions. Stop our own child after its
first new complete row and charge even native/controlled stops conservatively.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
PYTHON = 'D:/loopbase_env_20261025/Scripts/python.exe'
FROZEN = 'dbe4af336c619e5437ea31b9df91276ab125986b1a8d18d99bf85c0465eeebcf'
PREFIX = 'b2bb4dd03966cebaba23d32f7aac6981a5c58f16900841318e44da6e98d8efd9'
OUT = ROOT/'logs/native26_cold_context'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def put(path, value):
    temp = path.with_suffix(path.suffix+'.writing')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)

def live_owners():
    p = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
        "Get-CimInstance Win32_Process|Where-Object{$_.Name -eq 'python.exe' -and $_.CommandLine -match 'generate_math[.]py|actual_halting[.]py|run_suite[.]py|native26_boundary_replay[.]py'}|Select-Object ProcessId,CommandLine|ConvertTo-Json"],
        capture_output=True, text=True, check=True)
    return p.stdout.strip()

def prepare(trial):
    target = BASE/'extensions'/f'baseline_suite_20261025_seed2021_cold_{trial}'
    assert not target.exists(), 'Diagnostic root already exists; inspect instead of repeating'
    target.mkdir()
    names = ['scripts/common.py', 'scripts/methods.py', 'scripts/generate_math.py',
             'configs/protocol.json', 'data/math_validation.jsonl', 'data/math_fewshot.jsonl',
             'results/generation_parity_Ouro-2.6B.json', 'trajectories/math_Ouro-2.6B_validation.jsonl']
    manifest = {}
    for name in names:
        source, destination = ROOT/name, target/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        assert digest(source) == digest(destination), name
        manifest[name] = dict(bytes=source.stat().st_size, sha256=digest(source))
    files = [target/'scripts/common.py', target/'scripts/methods.py',
             BASE/'scripts/qwen_recipe.py', target/'scripts/generate_math.py']
    fingerprint = hashlib.sha256(''.join(p.name+digest(p) for p in files).encode()).hexdigest()
    assert fingerprint == FROZEN and manifest['trajectories/math_Ouro-2.6B_validation.jsonl']['sha256'] == PREFIX
    # A junction to the single authoritative budget prevents uncharged trials.
    escaped = str(target/'logs').replace("'", "''")
    actual = str(ROOT/'logs').replace("'", "''")
    subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                    f"New-Item -ItemType Junction -Path '{escaped}' -Target '{actual}'|Out-Null"], check=True)
    assert (target/'logs').is_junction()
    put(OUT/(trial+'_input_manifest.json'), dict(root=str(target), copied_files=manifest,
        generation_implementation_sha256=fingerprint,
        scope='Same producer bytes, protocol, inputs, checkpoint and module cache; isolated diagnostic outputs, authoritative shared GPU ledger'))
    return target

def main():
    assert str(BASE).replace('\\', '/').lower() == 'd:/looped_stage1_trajectory_audit_seed2021_v1'
    assert not live_owners(), 'An experiment GPU owner is live'
    assert not OUT.exists(), 'Cold-context diagnostic already started'
    OUT.mkdir()
    assert digest(ROOT/'trajectories/math_Ouro-2.6B_validation.jsonl') == PREFIX
    expected = json.loads((ROOT/'results/native26_recovery_next_case_probe.json').read_text())
    trials = []
    for trial, allocator in [('default1', None), ('debug1', 'debug')]:
        assert not live_owners()
        ledger = json.loads((ROOT/'logs/gpu_budget.json').read_text())
        initial_run_count = len(ledger['runs'])
        remaining = 50*3600-ledger['used_seconds']
        assert remaining > 420
        target = prepare(trial)
        env = dict(os.environ, PYTHONUTF8='1', PYTHONFAULTHANDLER='1',
                   HF_HOME=str(ROOT/'cache/huggingface'), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        if allocator is None:
            env.pop('PYTHONMALLOC', None)
        else:
            env['PYTHONMALLOC'] = allocator
        started, monotonic = time.time(), time.monotonic()
        reached = timed_out = False
        with (OUT/(trial+'_stdout.log')).open('w', encoding='utf-8') as out, (OUT/(trial+'_stderr.log')).open('w', encoding='utf-8') as err:
            child = subprocess.Popen([PYTHON, '-u', '-X', 'utf8', '-X', 'faulthandler',
                str(target/'scripts/generate_math.py'), '--model', 'Ouro-2.6B', '--stage', 'validation'],
                cwd=target, stdout=out, stderr=err, env=env)
            put(OUT/'OWNER.json', dict(controller_pid=os.getpid(), child_redirector_pid=child.pid,
                                      trial=trial, diagnostic_root=str(target), started_unix=started))
            while child.poll() is None:
                status_path = target/'STATUS.json'
                if status_path.exists():
                    try:
                        status = json.loads(status_path.read_text())
                        reached = status.get('completed', 0) >= 208
                    except (OSError, json.JSONDecodeError):
                        pass
                timed_out = time.monotonic()-monotonic > min(300, remaining-120)
                if reached or timed_out:
                    subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'], stdout=err, stderr=err)
                    child.wait()
                    break
                time.sleep(.1)
            code = child.wait()
        assert not live_owners(), 'A descendant is still live; do not charge/relaunch'
        elapsed = time.monotonic()-monotonic
        active = json.loads((ROOT/'logs/active_gpu.json').read_text())
        ledger = json.loads((ROOT/'logs/gpu_budget.json').read_text())
        # The child may have closed Budget for a normal Python exception. Native
        # and deliberate stops cannot; charge this complete controlled interval.
        previous_count = initial_run_count
        matching = [r for r in ledger['runs'][previous_count:]
                    if r.get('tag') == 'generation_Ouro-2.6B_validation']
        assert len(matching) <= 1
        if not matching:
            observed = (active.get('tag') == 'generation_Ouro-2.6B_validation'
                        and active.get('started_unix', 0) >= started-1)
            actual_pid = active['pid'] if observed else child.pid
            ledger['runs'].append(dict(tag='native26_cold_context_'+trial,
                pid=actual_pid, pid_role='actual_worker' if observed else 'redirector_before_budget',
                child_redirector_pid=child.pid,
                seconds=elapsed, state='diagnostic_controlled_stop' if reached else 'diagnostic_native_or_timeout',
                native_exit_code=code, python_allocator=allocator or 'default',
                accounting='Conservative entire diagnostic subprocess interval, including imports/startup/termination'))
            ledger['used_seconds'] += elapsed
            ledger['extension_used_seconds'] = ledger['used_seconds']-ledger['stage1_used_seconds']
            assert abs(ledger['used_seconds']-ledger['stage1_used_seconds']-sum(r['seconds'] for r in ledger['runs'])) < 1e-4
        else:
            actual_pid = matching[0]['pid']
            matching[0].update(diagnostic_trial=trial, diagnostic_root=str(target))
        put(ROOT/'logs/gpu_budget.json', ledger)
        active.update(pid=actual_pid)
        active.update(state='diagnostic_controlled_stop' if reached else 'failed',
                      finished_unix=time.time(), diagnostic_trial=trial)
        put(ROOT/'logs/active_gpu.json', active)
        raw = (target/'trajectories/math_Ouro-2.6B_validation.jsonl').read_bytes()
        assert hashlib.sha256(raw[:4795153]).hexdigest() == PREFIX
        boundary = raw.rfind(b'\n')+1
        rows = [json.loads(line) for line in raw[:boundary].splitlines()]
        next_row = next((r for r in rows if r['record_key'] == expected['record_key']), None)
        exact = next_row is not None and all(next_row[k] == expected[k]
                    for k in ['token_ids', 'text', 'prediction', 'correct', 'numeric_correct', 'condition'])
        argmax_exact = False
        if next_row is not None:
            assert next_row['implementation_sha256'] == FROZEN
            assert next_row['protocol_sha256'] == digest(ROOT/'configs/protocol.json')
            argmax_exact = ([[s['prediction'] for s in p['by_depth']] for p in next_row['token_trajectory']]
                          == [[s['prediction'] for s in p['by_depth']] for p in expected['token_trajectory']])
            put(OUT/(trial+'_next_case.json'), next_row)
        result = dict(trial=trial, allocator=allocator or 'default', diagnostic_root=str(target),
            native_exit_code=code, stopped_after_complete_next_row=reached, timeout=timed_out,
            complete_records=len(rows), next_record_exact=exact, baseline_argmax_exact=argmax_exact, elapsed_s=elapsed,
            cumulative_gpu_hours=ledger['used_seconds']/3600)
        trials.append(result)
        put(OUT/'RESULTS.json', dict(trials=trials, scope='Cold-context diagnostics only; deliberate stops are not native success exits and are not formal baseline results'))
        print(json.dumps(result), flush=True)
        assert digest(ROOT/'trajectories/math_Ouro-2.6B_validation.jsonl') == PREFIX
    put(OUT/'DONE.json', dict(completed=True, trials=trials, formal_prefix_unchanged=True,
                             scope='Diagnostic matrix completed; no formal validation progress or stable-runtime claim'))

if __name__ == '__main__':
    main()
