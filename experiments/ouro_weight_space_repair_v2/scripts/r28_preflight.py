"""Fixed24 request-only preparation/admission, no historical scores or gold."""
from r28_common import ROOT,read,sha,digest,save
from r28_prefix_task_inventory import selected,pair,Adapter

def build():
 from tokenizers import Tokenizer
 tokpath=ROOT/'artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json';assert sha(tokpath)=='fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d';tok=Adapter(Tokenizer.from_file(str(tokpath)));graph=read('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json');assert graph['status']=='passed_CPU_prefix_admission';cohort=graph['cohort'];wanted={r['id'] for r in cohort};source={}
 for split in ['discovery','selection']:
  with (ROOT/('data/slcd_'+split+'.jsonl')).open(encoding='utf8') as f:
   for line in f:
    identity=selected(line,{'id'})['id']
    if identity in wanted:source[identity]=selected(line,{'id','request_pairs','choice_char_lengths'})
 refs=read('configs/r7_reference_manifest.json')['references'];rows=[]
 for chosen in cohort:
  src=source[chosen['id']];candidates=[]
  for j,request in enumerate(src['request_pairs']):
   p,t=pair(tok,*request);rp={n:pair(tok,refs[n],request[1]) for n in ['N0','N1','N2']};record=next(r for r in graph['candidate_records'] if r['id']==chosen['id'] and r['candidate']==j)
   assert record['Q_target_positions']==len(t) and all(record['reference_position_maps'][n]==list(range(len(rp[n][1]))) and len(rp[n][1])==len(t) for n in rp),'Current pilot N0 full trajectory must be explicitly complete; future graphs need independent extra-position tasks'
   assert src['choice_char_lengths'][j]>0
   candidates.append(dict(candidate=j,request_pair=request,characters=src['choice_char_lengths'][j],Q_prefix_ids=p,Q_target_ids=t,reference_prefix_ids={n:x[0] for n,x in rp.items()},reference_target_ids={n:x[1] for n,x in rp.items()}))
  rows.append(dict(id=chosen['id'],split=chosen['split'],candidates=candidates))
 assert len(rows)==24 and sum(len(c['Q_target_ids']) for r in rows for c in r['candidates'])==548
 save('configs/r28_SANITIZED_REQUESTS.json',dict(rows=rows,scope='IDs/full candidate requests/original character denominator/tokenization only, no gold/predictions/scores',input_sha256={p:sha(p) for p in ['data/slcd_discovery.jsonl','data/slcd_selection.jsonl','configs/r7_reference_manifest.json','configs/r28_TRUE_PREFIX_TASK_INVENTORY.json']}))
 save('results/r28_PREFIX_CPU_ADMISSION.json',dict(status='passed',graph_sha256=sha('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json'),sanitized_requests_sha256=sha('configs/r28_SANITIZED_REQUESTS.json'),questions=24,candidates=96,Q_positions=548,N0_complete_positions=548,N1_complete_positions=548,N2_complete_positions=548,all96_reference_maps_identity=True,all_consumed_text_boundary_checks_passed=True,counts=graph['counts'],planning=graph['planning_heuristic'],model_loaded=False,GPU_worker_seconds=0))
 print('R28_SANITIZED_PREFLIGHT_PASS')
def audit_runtime(tok):
 graph=read('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json');rows=read('configs/r28_SANITIZED_REQUESTS.json')['rows'];from runtime import encode_pair
 import inspect
 from lm_eval.api.model import TemplateLM
 assert sha(inspect.getfile(TemplateLM))==graph['metadata']['lm_eval_source_sha256']
 for row in rows:
  for c in row['candidates']:
   p,t=encode_pair(tok,*c['request_pair']);assert p==c['Q_prefix_ids'] and t==c['Q_target_ids']
   for n,text in read('configs/r7_reference_manifest.json')['references'].items():
    if n in ['N0','N1','N2']:assert encode_pair(tok,text,c['request_pair'][1])==(c['reference_prefix_ids'][n],c['reference_target_ids'][n])
 for task in graph['tasks']:
  ids=graph['prefix_pool'][task['prefix_ref']]['ids']+task['consumed_ids'];assert digest(ids)==task['input_ids_sha256'] and len(ids)==task['length']
 return dict(status='passed_CPU_before_model',tasks=len(graph['tasks']),questions=24,N0_full_trajectory_verified=True,graph_sha256=sha('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json'),sanitized_sha256=sha('configs/r28_SANITIZED_REQUESTS.json'),model_loaded=False)
if __name__=='__main__':
 import sys
 if '--runtime' in sys.argv:
  from transformers import AutoTokenizer
  identity=read('configs/r27_EXECUTION_LOCK.json')['model_identity'];assert all(sha('models/Ouro-1.4B/'+f)==h for f,h in identity['metadata_and_tokenizer_sha256'].items())
  tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True);assert tok.is_fast;save('results/r28_RUNTIME_TOKENIZER_PREFLIGHT.json',audit_runtime(tok));print('R28_RUNTIME_TOKENIZER_PREFLIGHT_PASS')
 else:build()
