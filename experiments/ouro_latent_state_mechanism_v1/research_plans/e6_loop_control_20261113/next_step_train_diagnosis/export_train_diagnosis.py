"""Isolate Train-only fixture before any new fitting process starts."""
import json,hashlib
from pathlib import Path
R=Path('/root/autodl-tmp/ouro-e6a');O=R/'next_step_train_diagnosis';O.mkdir(exist_ok=True)
labels=json.loads(Path('/root/autodl-tmp/ouro-e4-e5/inputs/labels.json').read_text());rows=[]
for p in sorted((R/'joint/train').glob('train_*.json')):
 q=json.loads(p.read_text());l=labels['train:'+q['id']]
 if l['is_demonstration']:continue
 key=q['input_ids_sha256'];h=lambda s,n:int(hashlib.sha256((s+key).encode()).hexdigest(),16)%n
 rows.append({'id':q['id'],'split':'train','is_demonstration':False,'gold':l['gold'],'group':key,'outer_fold':h('ouro-e7-outer-20261113:',5),'inner_fold':h('ouro-e7-inner-20261113:',3),'input_ids_sha256':key,'capture_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scores':[n['scores'] for n in q['native']],'pred':[n['prediction'] for n in q['native']],'latent':[next(f['values'] for f in q['features'] if f['family']=='latent' and f['depth']==d) for d in range(1,5)],'unrelated':[next(f['values'] for f in q['features'] if f['family']=='unrelated' and f['depth']==d) for d in range(1,5)]})
assert len(rows)==1094
(O/'TRAIN_ONLY_INPUTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(O/'EXPORTER_RECEIPT.json').write_text(json.dumps({'scope':'Isolated Train fixture, no fitting. The subsequently executed fit process opens only this Train-only file, not the historical all-split labels.','n':len(rows),'exporter_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256((O/'TRAIN_ONLY_INPUTS.jsonl').read_bytes()).hexdigest()},indent=2))
print('TRAIN_ONLY_FIXTURE_READY',len(rows))
