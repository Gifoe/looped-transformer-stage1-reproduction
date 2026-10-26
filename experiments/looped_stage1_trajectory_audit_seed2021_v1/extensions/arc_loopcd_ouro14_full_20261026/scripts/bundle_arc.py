"""Immutable ARC-only evidence snapshot, independent of later GSM output."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
assert json.loads((ROOT/'logs/arc_controller/DONE.json').read_text())['completed']
names=['README.md','configs/protocol.json','data/arc_full_test.jsonl','data/arc_parity_validation.jsonl','data/FULL_MANIFEST_AUDIT.json',
       'sources/PAPER_PROTOCOL_EXTRACT.json','sources/harness_arc_easy.yaml','sources/harness_arc.yaml',
       'results/ARC_PARITY.json','results/ARC_WORKER_COMPLETE.json','results/ARC_REPLICATION_GATE.json',
       'results/model_load_Ouro-1.4B.json','results/arc_summary.csv','results/arc_paired_transitions.csv',
       'trajectories/arc_three_methods.jsonl','logs/arc_launcher.log','logs/arc_launcher_exit.json']
names += ['scripts/'+name for name in ['common.py','methods.py','windows_task_affinity.py','prepare_data.py','prepare_arc.py','score_arc.py','analyze_arc.py','arc_controller.py','arc_launcher.ps1','bundle_arc.py']]
names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'logs/arc_controller').iterdir() if p.is_file()]
names=sorted(set(names));manifest={name:dict(bytes=(ROOT/name).stat().st_size,sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()) for name in names}
destination=ROOT/'transport/arc_evidence.zip'
assert not destination.exists(),'Preserve the earlier immutable snapshot'
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in names:archive.write(ROOT/name,name)
    archive.writestr('ARC_EVIDENCE_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(destination) as archive:assert archive.testzip() is None
receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),bytes=destination.stat().st_size,
    sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),files=len(names),scope='Closed complete1172-question ARC reproduction only; no GSM completion claim')
(ROOT/'transport/arc_evidence_metadata.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
