"""Input provenance preflight; no GPU. Not a production driver."""
import json,hashlib
from pathlib import Path
from integrity import validate_records
ROOT=Path('/home/luansiyu/loop-latent')
ORIG=ROOT/'orchestration_runs/run-20261107T053919Z-6d01138e'
COVERAGE=ROOT/'orchestration_runs/resume-test-20261111/coverage_test.json'
SELECTION=ROOT/'research_plans/PREFIX_NATIVE_ABCA_SELECTION.json'
PREFIX=ROOT/'research_plans/natural_prefix_identity_20261111/reference_prefix_ids.json'
EXPECTED_SELECTION='1dcf857e8f9b05297fcb140a629c05f278707095de071747f62d2bc44d14cfac'
EXPECTED_COVERAGE='df164143c5921971c54a314ef560b3e3cc8566ff13636a931690efe53943d5e1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def inputs():
 if sha(PREFIX)!='e984a118a098923ba87fab4ddd4a26d1f9da8c5dc9767c5ab073352cd3dfbf67' or sha(SELECTION)!=EXPECTED_SELECTION or sha(COVERAGE)!=EXPECTED_COVERAGE:raise ValueError('frozen selection/coverage drift')
 sel=json.loads(SELECTION.read_text())['selection'];prefix=json.loads(PREFIX.read_text())['input_ids'];cov=json.loads(COVERAGE.read_text())
 if len(prefix)!=1839:raise ValueError('prefix length')
 questions=cov['questions']
 if len(questions)!=1172 or len({q['id'] for q in questions})!=1172 or any(q['status']!='complete' or q['errors'] for q in questions):raise ValueError('coverage incomplete')
 qs={q['id']:q for q in questions};out={}
 for name in ('A','B','C'):
  row=sel[name];ident=row['id'];q=qs[ident]
  if row['split']!='test' or q['split']!='test':raise ValueError('split')
  key=hashlib.sha256(ident.encode()).hexdigest();meta=ORIG/'shards/test'/(key+'.json');tensor=meta.with_suffix('.safetensors')
  if sha(meta)!=q['metadata_sha256'] or sha(tensor)!=q['tensor_sha256']:raise ValueError('natural shard hash')
  m=json.loads(meta.read_text());ids,positions,targets,scores=validate_records(m['records'],row,prefix)
  if q['actual_requests']!=q['expected_requests'] or len(targets)!=q['expected_requests']:raise ValueError('candidate count')
  out[name]={'id':ident,'ids':ids,'positions':positions,'targets':targets,'scores':scores,'metadata_sha256':sha(meta),'tensor_sha256':sha(tensor)}
 if sel['A']['input_length']!=sel['B']['input_length'] or out['A']['ids']==out['B']['ids'] or sel['C']['input_length']<sel['A']['input_length']:raise ValueError('selection structure')
 return out
if __name__=='__main__':
 raise SystemExit('NOT APPROVED: no CLI execution; provenance preflight requires review')
