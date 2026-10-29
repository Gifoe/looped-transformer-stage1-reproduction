"""CPU-only reconstruction and packaging after the frozen chain completes."""
from pathlib import Path
import hashlib
import json
import runpy
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT.parent / 'frozen_loop_update_value_seed2021_v1/scripts'))
import runtime_bootstrap


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    begin = time.monotonic()
    tools = load(ROOT / 'REPORT_TOOLS_MANIFEST.json')
    for rel, item in tools['files'].items():
        assert sha(ROOT / rel) == item['sha256'], rel
    runpy.run_path(str(ROOT / 'scripts/verify_completed.py'), run_name='__main__')
    prior = load(ROOT.parent / 'closed_loop_action_value_dev400_20261029/results/BUDGET.json')
    current = load(ROOT / 'results/BUDGET_RUNNING.json')
    charged = current['gpu_charge_seconds']
    budget = {'accounting': 'Whole GPU-worker intervals; CPU-only fitting, analysis and packaging excluded',
              'stage_gpu_seconds': charged, 'stage_gpu_hours': charged / 3600,
              'stage_cap_seconds': 10800, 'prior_new_research_seconds': prior['total_new_research_seconds'],
              'total_new_research_seconds': prior['total_new_research_seconds'] + charged,
              'remaining_original_8h_seconds': prior['remaining_original_8h_seconds'] - charged,
              'historical_cumulative_seconds': prior['cumulative_seconds'] + charged,
              'historical_budget_cap_seconds': 50 * 3600, 'historical_ledgers_modified': False}
    assert budget['remaining_original_8h_seconds'] >= 0 and budget['historical_cumulative_seconds'] <= 50 * 3600
    (ROOT / 'results/BUDGET_FINAL.json').write_text(json.dumps(budget, indent=2), encoding='utf8')
    runpy.run_path(str(ROOT / 'scripts/report_completed.py'), run_name='__main__')
    receipt = {'scope': 'CPU-only independent reconstruction, report rendering and packaging; figure visual inspection still pending',
               'completed_unix': time.time(), 'cpu_wall_seconds': time.monotonic() - begin,
               'new_gpu_seconds': 0, 'independent_final_evaluated': False,
               'source_lock_sha256': sha(ROOT / 'configs/LOCK.json')}
    (ROOT / 'results/FINALIZATION.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and 'transport' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts and p.suffix not in ['.zip', '.download', '.tmp'] and p.name != 'FINAL_ARTIFACT_MANIFEST.json')
    manifest = {'scope': 'Completed 3200-training/800-validation calibration, frozen sources/locks, raw outcomes, failed admission evidence and CPU reconstruction. Excludes transport, caches, bundle and self. PNG/PDF visual inspection recorded separately after download.',
                'files': {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files}}
    path = ROOT / 'FINAL_ARTIFACT_MANIFEST.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    archive = ROOT / 'delivery_bundle.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for p in files + [path]:
            bundle.write(p, p.relative_to(ROOT).as_posix())
    print(json.dumps({'payload_files': len(files), 'manifest_sha256': sha(path), 'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive), 'new_gpu_seconds': 0}), flush=True)


if __name__ == '__main__':
    main()
