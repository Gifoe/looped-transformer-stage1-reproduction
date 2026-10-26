"""Natural-exit diagnostic of all9 next-question conditions under process affinity.

The producer is byte-identical. Its declared24-question input prefix and isolated
outputs are diagnostic only; formal validation still requires100 questions.
"""
from pathlib import Path
import ctypes
from ctypes import wintypes
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
OUT = ROOT/'logs/native26_affinity_context'

KERNEL = ctypes.WinDLL('kernel32', use_last_error=True)
KERNEL.GetCurrentProcess.restype = wintypes.HANDLE
KERNEL.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
KERNEL.OpenProcess.restype = wintypes.HANDLE
KERNEL.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
KERNEL.GetProcessAffinityMask.restype = wintypes.BOOL
KERNEL.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
KERNEL.SetProcessAffinityMask.restype = wintypes.BOOL
KERNEL.CloseHandle.argtypes = [wintypes.HANDLE]

def affinity(pid=None):
    handle = KERNEL.GetCurrentProcess() if pid is None else KERNEL.OpenProcess(0x0400, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not KERNEL.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        return process.value, system.value
    finally:
        if pid is not None:
            KERNEL.CloseHandle(handle)

def restrict_this_controller():
    topology = json.loads((ROOT/'logs/native26_cpu_sets.json').read_text())
    cpus = topology['cpu_sets']
    assert all(c['group'] == 0 for c in cpus)
    classes = {c['efficiency_class'] for c in cpus}
    assert len(classes) > 1
    before, system = affinity()
    ids = [c['logical'] for c in cpus if c['efficiency_class'] == min(classes)
           and before & (1 << c['logical']) and not (c['flags'] & 2 and not c['flags'] & 4)]
    assert len(ids) >= 4
    mask = sum(1 << index for index in ids)
    assert mask & system == mask
    assert KERNEL.SetProcessAffinityMask(KERNEL.GetCurrentProcess(), mask), ctypes.get_last_error()
    assert affinity()[0] == mask
    put(OUT/'AFFINITY.json', dict(controller_pid=os.getpid(), before_mask=hex(before), mask=hex(mask),
        logical_cpu_ids=ids, efficiency_class=min(classes),
        scope='Only this diagnostic controller and inherited child processes; no other process, service, firmware, clock or system power setting changed'))
    return mask

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
    target = BASE/'extensions'/f'baseline_suite_20261025_seed2021_affinity_{trial}'
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
    # A declared diagnostic prefix lets the unmodified producer terminate
    # naturally after all9 conditions of the next question, with native exit0.
    name = 'data/math_validation.jsonl'
    original = (ROOT/name).read_bytes()
    lines = original.splitlines(keepends=True)
    assert len(lines) == 100
    subset = b''.join(lines[:24])
    (target/name).write_bytes(subset)
    manifest[name] = dict(bytes=len(subset), sha256=digest(target/name),
                         original_sha256=hashlib.sha256(original).hexdigest(),
                         diagnostic_prefix_questions=24, original_questions=100)
    put(target/'DIAGNOSTIC_ONLY.json', dict(original_validation_questions=100,
        diagnostic_prefix_questions=24, expected_diagnostic_records=216,
        formal_records_before_trial=207, formal_outputs_never_modified=True))
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
        scope='Identical producer/protocol/checkpoint/module cache, explicit byte-exact24-question validation prefix; diagnostic output only, authoritative shared GPU ledger'))
    return target

def main():
    assert str(BASE).replace('\\', '/').lower() == 'd:/looped_stage1_trajectory_audit_seed2021_v1'
    assert not live_owners(), 'An experiment GPU owner is live'
    assert not OUT.exists(), 'Cold-context diagnostic already started'
    OUT.mkdir()
    requested_mask = restrict_this_controller()
    assert digest(ROOT/'trajectories/math_Ouro-2.6B_validation.jsonl') == PREFIX
    expected = json.loads((ROOT/'results/native26_recovery_next_case_probe.json').read_text())
    trials = []
    for trial, allocator in [('ecore1', None)]:
        assert not live_owners()
        ledger = json.loads((ROOT/'logs/gpu_budget.json').read_text())
        initial_run_count = len(ledger['runs'])
        remaining = 50*3600-ledger['used_seconds']
        assert remaining > 1020
        target = prepare(trial)
        env = dict(os.environ, PYTHONUTF8='1', PYTHONFAULTHANDLER='1',
                   HF_HOME=str(ROOT/'cache/huggingface'), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        if allocator is None:
            env.pop('PYTHONMALLOC', None)
        else:
            env['PYTHONMALLOC'] = allocator
        started, monotonic = time.time(), time.monotonic()
        reached = timed_out = affinity_verified = False
        affinity_error = None
        with (OUT/(trial+'_stdout.log')).open('w', encoding='utf-8') as out, (OUT/(trial+'_stderr.log')).open('w', encoding='utf-8') as err:
            child = subprocess.Popen([PYTHON, '-u', '-X', 'utf8', '-X', 'faulthandler',
                str(target/'scripts/generate_math.py'), '--model', 'Ouro-2.6B', '--stage', 'validation'],
                cwd=target, stdout=out, stderr=err, env=env)
            put(OUT/'OWNER.json', dict(controller_pid=os.getpid(), child_redirector_pid=child.pid,
                                      trial=trial, diagnostic_root=str(target), started_unix=started))
            while child.poll() is None:
                if not affinity_verified:
                    active_path = ROOT/'logs/active_gpu.json'
                    current = json.loads(active_path.read_text())
                    if current.get('started_unix', 0) >= started-1 and current.get('tag') == 'generation_Ouro-2.6B_validation':
                        try:
                            actual_mask = affinity(current['pid'])[0]
                            affinity_verified = actual_mask == requested_mask
                            if not affinity_verified:
                                affinity_error = f'Unexpected worker mask {hex(actual_mask)}'
                                subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'], stdout=err, stderr=err)
                                child.wait()
                                break
                            put(OUT/(trial+'_child_affinity.json'), dict(pid=current['pid'], mask=hex(actual_mask), verified=True))
                        except OSError:
                            pass
                status_path = target/'STATUS.json'
                if status_path.exists():
                    try:
                        status = json.loads(status_path.read_text())
                        reached = status.get('completed', 0) >= 216
                    except (OSError, json.JSONDecodeError):
                        pass
                timed_out = time.monotonic()-monotonic > min(900, remaining-120)
                if timed_out:
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
            ledger['runs'].append(dict(tag='native26_affinity_context_'+trial,
                pid=actual_pid, pid_role='actual_worker' if observed else 'redirector_before_budget',
                child_redirector_pid=child.pid,
                seconds=elapsed, state='diagnostic_native_or_timeout',
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
        active.update(state='finished' if code==0 else 'failed',
                      finished_unix=time.time(), diagnostic_trial=trial)
        put(ROOT/'logs/active_gpu.json', active)
        raw = (target/'trajectories/math_Ouro-2.6B_validation.jsonl').read_bytes()
        assert hashlib.sha256(raw[:4795153]).hexdigest() == PREFIX
        boundary = raw.rfind(b'\n')+1
        rows = [json.loads(line) for line in raw[:boundary].splitlines()]
        if code == 0:
            docs = [json.loads(line) for line in (target/'data/math_validation.jsonl').read_bytes().splitlines()]
            methods = json.loads((ROOT/'configs/protocol.json').read_text())['math']['validation_methods']
            assert len(rows) == 216 and boundary == len(raw)
            assert {r['record_key'] for r in rows} == {d['id']+'|'+m for d in docs for m in methods}
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
            native_exit_code=code, all_next_question_conditions_completed=reached, timeout=timed_out,
            complete_records=len(rows), next_record_exact=exact, baseline_argmax_exact=argmax_exact, elapsed_s=elapsed,
            child_affinity_verified=affinity_verified, affinity_mask=hex(requested_mask),
            affinity_error=affinity_error,
            cumulative_gpu_hours=ledger['used_seconds']/3600)
        trials.append(result)
        put(OUT/'RESULTS.json', dict(trials=trials, scope='Process-only CPU-affinity diagnostic on a declared24-question input prefix, with23 questions already complete; not formal100-question validation'))
        print(json.dumps(result), flush=True)
        assert digest(ROOT/'trajectories/math_Ouro-2.6B_validation.jsonl') == PREFIX
        assert code == 0 and affinity_verified and len(rows) == 216 and exact and argmax_exact
    put(OUT/'DONE.json', dict(completed=True, trials=trials, formal_prefix_unchanged=True,
                             scope='Affinity diagnostic completed; its24-question prefix is not the formal100-question validation and is never merged into formal trajectories'))

if __name__ == '__main__':
    main()
