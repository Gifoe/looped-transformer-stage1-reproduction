"""Regression: summary failure, missing candidate, stale data, duplicate ID."""
import tempfile,json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from manifest_guard import validate_manifest,file_sha
ROOT=Path(__file__).resolve().parents[3]
with tempfile.TemporaryDirectory(prefix='manifest_guard_',dir=ROOT) as d:
 d=Path(d);data=d/'data.parquet';manifest=d/'manifest.jsonl';summary=d/'summary.json';rows=[{'id':'x','choices':{'text':['yes','no'],'label':['1','2']},'answerKey':'1'}];pq.write_table(pa.Table.from_pylist(rows),data)
 records=[{'id':'x','candidate':i,'split':'validation','option_count':2,'raw_labels':['1','2'],'answer_key':'1','target_token_count':1,'input_length':100,'selected_token_count':2} for i in range(2)]
 def save(rs,errors=()):
  manifest.write_text(''.join(json.dumps(r)+'\n' for r in rs));s={'splits':{'validation':{'errors':list(errors),'expected_questions':1,'unique_questions':1,'data_sha256':file_sha(data),'manifest_sha256':file_sha(manifest)}}};summary.write_text(json.dumps(s))
 def run():return validate_manifest('validation',data,summary,manifest)
 def rejects():
  try:run()
  except Exception:return
  raise AssertionError('invalid manifest accepted')
 save(records);assert run()['requests']==2
 save(records,['some failure']);rejects()
 save(records[:1]);rejects() # hash recomputed; still catches missing candidate
 save([records[0],records[0]]);rejects()
 save(records);data.write_bytes(data.read_bytes()+b'x');rejects()
print('PASS failed summary, missing candidate, duplicate request and stale dataset')
