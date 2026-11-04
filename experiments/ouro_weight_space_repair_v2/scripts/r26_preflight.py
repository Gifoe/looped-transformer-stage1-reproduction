"""CPU tokenizer-only input selection before any model load."""
from r26_common import ROOT,read,jsonlines,save,digest
from r26_protocol import build_inputs

def audit(tok):
 old=read('configs/r25_HELLINGER_MEMORY_LOCK.json');rows={r['id']:r for s in ['discovery','selection'] for r in jsonlines('data/slcd_'+s+'.jsonl')}
 p=build_inputs(tok,old['cohort'],rows,read('configs/r7_reference_manifest.json')['references'],read('configs/r25_MEMORY_MANIFEST.json'))
 previous=read('results/r25_INPUT_AUDIT.json');r=next(r for r in previous['records'] if r['id']==p['id'] and r['candidate']==p['candidate'])
 from runtime import encode_pair
 q,t=encode_pair(tok,*rows[p['id']]['request_pairs'][p['candidate']]);assert digest(q)==r['prefix_ids_sha256'] and digest(t)==r['target_ids_sha256']
 manifest=read('configs/r24_INPUT_MANIFEST.json');m=next(x for x in manifest['rows'] if x['id']==p['id']);j=p['candidate']
 assert m['request_pairs'][j]==rows[p['id']]['request_pairs'][j] and m['prompt']==rows[p['id']]['request_pairs'][j][0]
 assert m['target_id_sha256'][j]==digest(t) and m['choice_token_counts'][j]==len(t)
 assert m['choice_char_lengths'][j]==rows[p['id']]['choice_char_lengths'][j]
 state=next(x for x in read('results/slcd_SOURCE_AUDIT.json')['states'] if x['id']==p['id']);assert state['target_id_sha256'][j]==digest(t)
 return dict(status='passed_CPU_pre_model',plan=p,plan_sha256=digest(p),Q_original_R25_prefix_target_hashes_exact=True,all143_offsets_values_predecessors_exact=True,shared_key_prefix_exact=True,source='Original R25 cohort/order and independently encoded Q/N012; no labels/predictions',model_loaded=False,GPU_worker_seconds=0)
if __name__=='__main__':
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True);assert tok.is_fast
 save('results/r26_CPU_INPUT_PREFLIGHT.json',audit(tok));print('R26_CPU_INPUT_PREFLIGHT_PASS')
