"""One CPU-only rebuild command; missing experiments are errors, never skipped."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
MODELS = ['Ouro-1.4B', 'Ouro-2.6B']


def steps(scope):
    result = [
        ('protocol_tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v']),
        ('tokenizer_exports', ['scripts/verify_audit_tokenizers.py']),
        ('statistics', ['scripts/analyze_suite.py', '--stage', scope]),
        ('mc_diagnostics', ['scripts/analyze_diagnostics.py']),
        ('mc_reference', ['scripts/compare_mc_rebuild.py']),
    ]
    if scope == 'mc':
        result.append(('mc_audit', ['scripts/audit_suite.py', '--stage', 'mc']))
    else:
        result.extend([
            ('latency_statistics', ['scripts/analyze_latency.py']),
            ('validation_integrity', ['scripts/audit_math_prefix.py', '--model', 'all', '--require-complete']),
            ('recovery_lineage_1p4', ['scripts/verify_recovery_lineage.py']),
            ('recovery_lineage_2p6', ['scripts/verify_native26_recovery_lineage.py']),
            ('actual_halting_integrity', ['scripts/audit_actual_halting.py', '--model', 'all']),
            ('historical_windows', ['scripts/audit_historical_windows.py']),
            ('completion_audit', ['scripts/audit_suite.py']),
            ('report', ['scripts/write_report.py']),
        ])
    return result


def preflight(scope):
    required = [BASE/'ARTIFACT_MANIFEST.json', ROOT/'configs/protocol.json',
                ROOT/'audit_reference/mc/REFERENCE_PROVENANCE.json',
                ROOT/'sources/tokenizer_export_manifest.json']
    if scope == 'all':
        required += [ROOT/'configs'/name for name in ['research_decision.json', 'math_selected.json', 'math_execution.json']]
        for model in MODELS:
            required += [ROOT/f'trajectories/math_{model}_{split}.jsonl' for split in ['validation', 'test']]
            required += [ROOT/f'trajectories/actual_halting_{model}.jsonl', ROOT/f'results/actual_halting_parity_{model}.json']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError('Required completed inputs are missing; no steps were run: '+json.dumps(missing))
    # Explicitly force CPU visibility even if this command runs on a GPU host.
    # None of the permitted step scripts load a model or invoke GPU producers.
    assert all(not any(Path(arg).name == producer for arg in args) for _, args in steps(scope)
               for producer in ['generate_math.py', 'score_mc.py', 'actual_halting.py', 'run_suite.py'])


def write(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['all', 'mc'], default='all',
                        help='all requires all experiments and the research decision; mc only certifies MC rebuilding')
    args = parser.parse_args()
    preflight(args.scope)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'logs/cpu_rebuild'/timestamp
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(passed=False, scope=args.scope, python=sys.version, executable=sys.executable,
                   started_utc=timestamp, cpu_only=True, steps=[],
                   entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   scope_note='MC scope does not certify math, actual stopping, full completion or final delivery' if args.scope == 'mc'
                              else 'Full CPU rebuild verifies supplied outputs; it does not rerun GPU inference or certify author-code parity')
    destination = output/'RECEIPT.json'
    write(destination, receipt)
    env = dict(os.environ, PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for label, arguments in steps(args.scope):
        command = [sys.executable, '-u', '-X', 'utf8', '-X', 'faulthandler', *arguments]
        started = time.monotonic()
        print(json.dumps(dict(step=label, state='running')), flush=True)
        with (output/(label+'.log')).open('w', encoding='utf-8') as stream:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = dict(name=label, command=command, exit_code=process.returncode,
                      elapsed_s=time.monotonic()-started, log=str(output/(label+'.log')))
        receipt['steps'].append(record)
        write(destination, receipt)
        if process.returncode:
            print(json.dumps(dict(passed=False, failed_step=label, receipt=str(destination))), flush=True)
            return process.returncode
        print(json.dumps(dict(step=label, state='passed')), flush=True)
    receipt.update(passed=True, finished_utc=datetime.now(timezone.utc).isoformat())
    write(destination, receipt)
    print(json.dumps(dict(passed=True, scope=args.scope, receipt=str(destination))), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
