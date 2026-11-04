"""R25 post-run descriptive inventory/accounting; no new query/key/model run."""
import json,collections,numpy as np
from r25_common import ROOT,read,sha,save

def main():
 s=read('configs/r25_HELLINGER_MEMORY_LOCK.json');a=read('results/r25_ANALYSIS.json');d=read('results/r25_COMPLETE.json');u=read('logs/r25_SUPERVISOR.json');b=read('logs/r25_gpu_budget.json');m=read('configs/r25_MEMORY_MANIFEST.json');rows=[json.loads(x) for x in (ROOT/'results/r25_pilot.jsonl').read_text().splitlines()];assert len(rows)==24 and d['all_parameter_hashes_before']==d['all_parameter_hashes_after'] and u['status']=='complete'
 assert all(sha('scripts/'+p)==h for p,h in s['source_sha256'].items()) and all(sha(p)==h for p,h in s['budget']['inherited_ledger_sha256'].items())
 occurrence={k:collections.Counter() for k in ['Q','N0','N1','N2']};value={k:collections.Counter() for k in occurrence};gaps=[];same_occurrence=0;same_value=0;comparisons=0;raw_collisions=0
 for row in rows:
  for c in row['candidates']:
   retrieval=c['retrieval'];q=retrieval[0]['neighbor_occurrences']
   for r in retrieval:
    occurrence[r['context']].update(r['neighbor_occurrences']);value[r['context']].update(m['values'][i] for i in r['neighbor_occurrences']);gaps.extend(r['minimum_two_distance_gap'])
   for r in retrieval[1:]:
    for x,y in zip(q,r['neighbor_occurrences']):same_occurrence+=x==y;same_value+=m['values'][x]==m['values'][y];comparisons+=1
   raw_collisions+=sum(not x and y for x,y in zip(c['metrics']['c_zero'],c['metrics']['raw_c_zero']))
 total=sum(read(p)['used_seconds'] for p in s['budget']['inherited_ledger_sha256'])+b['used_seconds'];assert total==b['total_used_seconds']
 out=dict(status='complete24_postrun_inventory',original_strong_screen_passed=a['original_strong_screen_passed'],cost_review_eligible=a['cost_review_eligible'],neighbor_occurrence_occupancy={k:dict(v) for k,v in occurrence.items()},neighbor_value_occupancy={k:dict(v) for k,v in value.items()},query_reference_same_occurrence=same_occurrence,query_reference_same_value=same_value,query_reference_comparisons=comparisons,query_reference_same_value_rate=same_value/comparisons,distance_gap_min=float(np.min(gaps)),distance_gap_median=float(np.median(gaps)),raw_null_collapses_nonzero_main_c_positions=raw_collisions,main_zero_c_positions=a['label_free_integrity']['zero_c_positions'],positions=a['label_free_integrity']['positions'],prediction_disagreements={k:sum(r['primary_prediction']!=r[k+'_prediction'] for r in a['question_results']) for k in a['counts'] if k!='primary'},worker_seconds=b['used_seconds'],component_seconds=d['component_seconds'],model_load_seconds=d['model_load_seconds'],aggregate_worker_seconds=total,remaining_worker_seconds=21600-total,peak_allocator_bytes=d['peak_cuda_allocated_bytes'],protocol_sha256=d['protocol_sha256'],raw_sha256=sha('results/r25_pilot.jsonl'),no_new_feature_or_model_evaluation=True)
 save('results/r25_POSTRUN_INVENTORY.json',out);print(json.dumps({k:out[k] for k in ['prediction_disagreements','query_reference_same_value_rate','main_zero_c_positions','raw_null_collapses_nonzero_main_c_positions','aggregate_worker_seconds','remaining_worker_seconds']}))
if __name__=='__main__':main()
