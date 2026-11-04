"""Post-run CPU support table consistency; no tokenizer/model/new supports."""
import json
from r24_common import ROOT,read,sha,save
from r24_support import candidate_support

def main():
 s=read('configs/r24_MEMORY_SUPPORT_LOCK.json');d=read('results/r24_SUPPORT_COMPLETE.json');u=read('logs/r24_SUPERVISOR.json');a=read('results/r24_ADMISSION.json');v=read('results/r24_VALUE_INVENTORY.json');m=read('configs/r24_INPUT_MANIFEST.json');rows=[json.loads(x) for x in (ROOT/'results/r24_support.jsonl').read_text().splitlines()]
 assert len(rows)==d['n']==576 and d['status']=='complete_gold_free_support' and u['status']=='complete' and not u['overrun']
 assert [r['id'] for r in rows]==[r['id'] for r in m['rows']] and d['support_sha256']==sha('results/r24_support.jsonl') and d['value_inventory_sha256']==sha('results/r24_VALUE_INVENTORY.json')
 assert all(sha('scripts/'+p)==h for p,h in s['source_sha256'].items())
 for row,original in zip(rows,m['rows']):
  assert row['R6_winner']==original['R6_winner'] and row['audit']['prompt_structured_bytes_exact'] and row['audit']['offsets_match_actual_encode_pair_prefix_IDs'] and row['audit']['target_ids_and_candidate_order_exact']
  assert candidate_support([c['target_set'] for c in row['candidates']],row['R6_winner'],set(v['S_all']),set(v['S_specific']))==row['candidates']
  assert row['candidates'][row['R6_winner']]['specific_difference_vs_R6']==[] and row['audit']['value_inventory_sha256']==d['value_inventory_sha256']
 assert all(e['source_token_position']>0 and e['key_prediction_position']==e['source_token_position']-1 for e in v['entries'])
 save('results/r24_EXECUTION_VERIFIED.json',dict(status='complete576_support_audits_admission_passed',all576_set_consistency=True,R6_anchor_difference_always_empty=True,value_occurrences=len(v['entries']),boundary_tokens=len(v['boundaries']),S_all_types=len(v['S_all']),S_specific_types=len(v['S_specific']),specific_demo_count=len(v['specific_demo_ids']),admission_passed=a['admission_passed'],CPU_supervised_seconds=u['CPU_wall_seconds'],GPU_worker_seconds=0,model_state_key_query_loaded=False,protocol_sha256=d['protocol_sha256'],support_sha256=d['support_sha256'],scope='Consistency replay only, not independent tokenizer extraction or semantic inference'))
 print('R24_POSTRUN_CONSISTENCY_PASS')
if __name__=='__main__':main()
