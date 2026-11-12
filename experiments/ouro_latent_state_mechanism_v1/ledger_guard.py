"""Only complete same-GPU exact-score replay may admit analysis."""
import hashlib
from pathlib import Path

def require_production_ledger(ledger):
 if ledger.get('ledger_schema')!='ouro-coverage-v2' or ledger.get('ledger_guard_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
  raise RuntimeError('stale coverage ledger schema or guard')
 path=Path(__file__).with_name('audit_coverage.py')
 if ledger.get('audit_source_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
  raise RuntimeError('stale or missing coverage audit source')
 replay=ledger.get('score_replay',{})
 if replay.get('device')!='cuda' or replay.get('status')!='passed' or replay.get('tolerance')!=0.0 or not replay.get('gpu') or not replay.get('cuda'):
  raise RuntimeError('complete exact CUDA score-replay ledger required')
