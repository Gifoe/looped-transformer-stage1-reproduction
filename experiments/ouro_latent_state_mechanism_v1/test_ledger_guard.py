import hashlib
from pathlib import Path
from ledger_guard import require_production_ledger
base=Path(__file__).parent;sha=lambda p:hashlib.sha256((base/p).read_bytes()).hexdigest()
def good():return {'ledger_schema':'ouro-coverage-v2','audit_source_sha256':sha('audit_coverage.py'),'ledger_guard_sha256':sha('ledger_guard.py'),'score_replay':{'device':'cuda','status':'passed','tolerance':0.0,'gpu':'synthetic','cuda':'synthetic'}}
def rejects(value):
 try:require_production_ledger(value)
 except RuntimeError:return
 raise AssertionError('invalid ledger accepted')
x=good();require_production_ledger(x) # unit fixture, not an actual GPU result
for field in ('ledger_schema','audit_source_sha256','ledger_guard_sha256'):
 x=good();x[field]='stale';rejects(x)
x=good();x['score_replay']['device']='cpu';rejects(x)
x=good();x['score_replay']['status']='incomplete';rejects(x)
print('PASS stale schema/source/guard, CPU and incomplete ledger rejection')
