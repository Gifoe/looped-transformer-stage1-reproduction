"""Small diagnostic evidence archive; excludes weights and formal-output changes."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT.parent/(ROOT.name+'_affinity_ecore1')
OUT = ROOT/'logs/native26_affinity_context'
assert json.loads((OUT/'DONE.json').read_text())['completed']
assert json.loads((ROOT/'logs/native26_affinity_context_exit.json').read_text())['exit_code'] == 0
items = {p.relative_to(ROOT).as_posix(): p for p in OUT.iterdir() if p.is_file()}
for name in ['logs/native26_affinity_context_exit.json', 'logs/gpu_budget.json']:
    items[name] = ROOT/name
for p in (ROOT/'logs/native26_bytecode_audit').rglob('*'):
    if p.is_file():
        items[p.relative_to(ROOT).as_posix()] = p
for name in ['data/math_validation.jsonl', 'trajectories/math_Ouro-2.6B_validation.jsonl', 'DIAGNOSTIC_ONLY.json', 'STATUS.json']:
    items['diagnostic/'+name] = DIAG/name
manifest = {name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for name, p in items.items()}
destination = ROOT/'transport/native26_affinity_evidence.zip'
assert not destination.exists(), 'Preserve an existing snapshot instead of overwriting it'
with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, p in items.items():
        archive.write(p, name)
    archive.writestr('EVIDENCE_MANIFEST.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
receipt = dict(bytes=destination.stat().st_size, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), files=len(items))
(ROOT/'transport/native26_affinity_evidence_metadata.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
