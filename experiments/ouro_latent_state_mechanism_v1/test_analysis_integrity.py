"""A ledger must not admit metadata changed after coverage validation."""
import tempfile,json,hashlib,sys
from pathlib import Path
import behavior_summary as b
root=b.PROJECT
with tempfile.TemporaryDirectory(prefix='analysis_guard_',dir=root) as d:
 d=Path(d);row={'id':'fixture','answerKey':'A','choices':{'label':['A','B'],'text':['x','y']}};key=hashlib.sha256(b'fixture').hexdigest();payload=d/(key+'.safetensors');payload.write_bytes(b'fixture');meta=d/(key+'.json');data={'id':'fixture','split':'validation','sha256':{'fixture':'hash'},'tensor_sha256':b.sha(payload),'gold_index':0,'predictions':[0]*4,'records':[{'candidate':0,'scores':[1]*4},{'candidate':1,'scores':[0]*4}]}
 meta.write_text(json.dumps(data));ledger={'counts':{'expected_questions':1,'complete_questions':1,'expected_requests':2,'complete_requests':2,'failed_questions':0,'missing_questions':0},'questions':[{'id':'fixture','split':'validation','status':'complete','tensor_sha256':b.sha(payload),'metadata_sha256':b.sha(meta)}]};coverage=d/'coverage.json';coverage.write_text(json.dumps(ledger));data['records'].pop();meta.write_text(json.dumps(data))
 b.require_production_ledger=lambda _:None # synthetic fixture only
 b.bound_hashes=lambda split:{'fixture':'hash'};b.pq.read_table=lambda path:type('T',(),{'to_pylist':lambda self:[row]})()
 sys.argv=['behavior_summary.py','--split','validation','--shards',str(d),'--coverage',str(coverage),'--out',str(d/'out.json')]
 try:b.main()
 except RuntimeError as exc:assert 'stale or corrupt' in str(exc)
 else:raise AssertionError('modified metadata was accepted')
print('PASS post-audit candidate removal rejected')
