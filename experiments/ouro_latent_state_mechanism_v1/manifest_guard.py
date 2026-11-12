"""Fail closed on input audit failures, missing requests or stale source."""
import hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def validate_manifest(split,data_path,summary_path,manifest_path):
 data_path,summary_path,manifest_path=map(Path,(data_path,summary_path,manifest_path));summary=json.loads(summary_path.read_text(encoding='utf8'))['splits'][split];rows=pq.read_table(data_path).to_pylist();ids=[r['id'] for r in rows]
 if len(ids)!=len(set(ids)):raise RuntimeError('duplicate data IDs')
 if summary['errors'] or summary['expected_questions']!=len(rows) or summary['unique_questions']!=len(rows):raise RuntimeError('input manifest reports failed or incomplete questions')
 if summary['data_sha256']!=file_sha(data_path) or summary['manifest_sha256']!=file_sha(manifest_path):raise RuntimeError('stale manifest or dataset')
 expected={(r['id'],i) for r in rows for i in range(len(r['choices']['text']))};data_by_id={r['id']:r for r in rows};seen=set()
 with manifest_path.open(encoding='utf8') as f:
  for line in f:
   r=json.loads(line);key=(r['id'],r['candidate'])
   if r['split']!=split or key in seen or key not in expected or r['target_token_count']!=1 or not 1<=r['input_length']<=32768 or r['selected_token_count']<1:raise RuntimeError('invalid, duplicate, or unsupported request')
   source=data_by_id[r['id']]
   if r['option_count']!=len(source['choices']['text']) or r['raw_labels']!=source['choices']['label'] or r['answer_key']!=source['answerKey']:raise RuntimeError('label or option metadata mismatch')
   seen.add(key)
 if seen!=expected:raise RuntimeError('missing request in input manifest')
 return {'questions':len(rows),'requests':len(seen)}
