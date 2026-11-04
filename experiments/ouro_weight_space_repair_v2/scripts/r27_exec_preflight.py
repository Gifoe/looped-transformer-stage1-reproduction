"""Frozen synthetic inputs/tokenizer metadata only; before model load."""
from r27_exec_common import ROOT,read,sha,digest,save
from r27_exec_protocol import ordered_instances

def audit(tok):
 manifest=read('configs/r27_CPU_STRUCTURE_MANIFEST.json');rows=ordered_instances(manifest)
 from runtime import encode_pair
 records=[]
 for r in rows:
  encoded=tok(r['prompt'],add_special_tokens=False,return_offsets_mapping=True);ids=encoded['input_ids'];assert ids==r['input_ids'] and r['input_shape']==[1,len(ids)] and digest(ids)==r['input_ids_sha256']
  for field in r['sources']:
   ix=field['position'];assert list(encoded['offset_mapping'][ix])==field['offset'] and ids[ix]==field['token_id']
   start,end=field['field_start'],field['field_end'];overlap=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>a and a<end and b>start];assert overlap==[ix] and start<=field['offset'][0]<field['offset'][1]<=end
   p,t=encode_pair(tok,r['prompt'],' '+field['label']);assert p==ids and t==field['continuation_ids']==[field['token_id']]
  assert r['source_positions']==sorted(set(r['source_positions'])) and len(r['source_positions'])==4 and max(r['source_positions'])<len(ids)-1
  records.append(dict(table=r['table'],query_row=r['query_row'],version=r['version'],input_ids_sha256=digest(ids),source_positions=r['source_positions'],input_shape=r['input_shape']))
 assert len(records)==8
 return dict(status='passed_tokenizer_pre_model',manifest_sha256=sha('configs/r27_CPU_STRUCTURE_MANIFEST.json'),records=records,model_loaded=False,GPU_worker_seconds=0)
if __name__=='__main__':
 from transformers import AutoTokenizer
 identity=read('configs/r26_CAUSAL_ACCEPTANCE_LOCK.json')['model_identity']
 assert all(sha('models/Ouro-1.4B/'+f)==h for f,h in identity['metadata_and_tokenizer_sha256'].items())
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True);assert tok.is_fast
 save('results/r27_EXEC_INPUT_PREFLIGHT.json',audit(tok));print('R27_EXEC_INPUT_PREFLIGHT_PASS')
