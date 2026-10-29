"""Checksum-bound recovery from a Windows progress-file sharing violation.

No scorer, policy, model or frozen input changes. The existing scorer replays
the final saved question before appending any new result.
"""
import json,hashlib,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
raw=ROOT/'results/final_piqa.jsonl'
with raw.open(encoding='utf8') as f:rows=[json.loads(l) for l in f if l.strip()]
with (ROOT/'data/final_piqa.jsonl').open(encoding='utf8') as f:source=[json.loads(l) for l in f if l.strip()]
receipt=json.loads((ROOT/'logs/final_piqa_exit.json').read_text())
error=(ROOT/'logs/final_piqa.err').read_text(encoding='utf8',errors='replace')
lock=json.loads((ROOT/'configs/FINAL_LOCK.json').read_text())
assert receipt['exit_code']==1 and 'PermissionError' in error and 'STATUS.tmp' in error
assert 0<len(rows)<len(source)==1802
assert [r['id'] for r in rows]==[r['id'] for r in source[:len(rows)]]
assert all(r['gold']==s['gold'] for r,s in zip(rows,source))
signature=rows[0]['signature'];assert all(r['signature']==signature for r in rows)
assert signature['score.py']==sha(ROOT/'scripts/score.py')
assert signature['protocol.json']==sha(ROOT/'configs/protocol.json')
assert signature['final_piqa.jsonl']==sha(ROOT/'data/final_piqa.jsonl')
for rel,h in lock['frozen_files'].items():assert sha(ROOT/rel)==h
archive=ROOT/'results/final_piqa_attempt1.jsonl';assert not archive.exists();shutil.copy2(raw,archive)
for name in ['final_piqa.err','final_piqa.out','final_piqa_exit.json','final_launcher.log']:
 p=ROOT/'logs'/name;dest=p.with_name('attempt1_'+name);assert not dest.exists();shutil.copy2(p,dest)
record=dict(partial_sha256=sha(raw),completed=len(rows),previous_signature=signature,approved_unix=time.time(),reason='Windows progress-file atomic replacement failed after a complete JSONL record was flushed. Scorer and frozen protocol remain byte-identical. Monitoring now uses FileShare.Delete; resume replays the boundary question with the existing 1e-5 score tolerance.')
(ROOT/'configs/RESUME_final_piqa.json').write_text(json.dumps(record,indent=2),encoding='utf8')
(ROOT/'results/PIQA_OPERATIONAL_RECOVERY.json').write_text(json.dumps(dict(passed=True,**record,original_exit=receipt),indent=2),encoding='utf8')
print(json.dumps(record),flush=True)
