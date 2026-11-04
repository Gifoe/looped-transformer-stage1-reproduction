"""R25 two-pass analysis: exhaustive label-free integrity BEFORE any labels/argmax."""
import math,numpy as np
from r25_common import ROOT,LOCK,read,sha,save,gate,jsonlines
from r19_analyze import paired
NAMES=['primary','M3','M2','q_only','reference_only','memory_direction_only','frequency','raw_occurrence_shuffle','freq_bijection_matched','leaveout_0','leaveout_1','leaveout_2']
METRICS={'M3_alpha','M2_alpha','c_zero','raw_c_zero','tau_c_zero','c_nonzero_coordinates','main_P_norm2','matched_P_norm2','unmatched','query_reference_same_value'}

def integrity(s,d,u,ka,rows):
 # This function reads sanitized metadata only, never current gold/old result rows.
 assert d['status']==u['status']=='complete' and d['n']==24 and u['exit_code']==0 and not u['overrun'] and not u.get('timed_out',False) and 0<=u['gpu_worker_seconds']<=300
 assert d['all_parameter_hashes_before'] and d['all_parameter_hashes_before']==d['all_parameter_hashes_after'] and not d['weights_modified']
 assert d['protocol_sha256']==sha(LOCK) and d['score_file_sha256']==sha('results/r25_pilot.jsonl') and d['key_audit_sha256']==sha('results/r25_KEY_AUDIT.json') and d['input_audit_sha256']==sha('results/r25_INPUT_AUDIT.json')
 assert [r['id'] for r in rows]==[r['id'] for r in s['cohort']] and len(rows)==24
 memory=read('configs/r25_MEMORY_MANIFEST.json');metadata={r['id']:r for r in read('configs/r24_INPUT_MANIFEST.json')['rows']};ia=read('results/r25_INPUT_AUDIT.json');assert ia['status']=='passed_CPU_all24_pre_model_input_audit' and ia['n']==24 and ia['all143value_ids_offsets_predecessors_exact'] and not ia['model_or_key_or_query_loaded']
 authority=read('configs/r25_PARITY_AUTHORITY.json')['rows']
 expected_positions=[e['key_prediction_position'] for e in memory['entries']];assert ka['selected_vs_full_native4_BF16_states_exact'] and ka['selected_vs_full_native4_head_fullV_exact'] and ka['positions']==expected_positions and ka['key_count']==143 and ka['head_shape']==[143,memory['vocabulary_size']] and ka['attention_backend']=='sdpa' and ka['key_input_ids_sha256']==ia['key_input_ids_sha256']
 assert ka['sqrt_key_bytes']==143*memory['vocabulary_size']*8 and ka['original_uncached_run_source_sha256']==sha('scripts/ouro_cached_scoring.py')
 input_records={(r['id'],r['candidate']):r for r in ia['records']};assert len(input_records)==ia['candidates']==sum(len(metadata[r['id']]['request_pairs']) for r in rows)
 book=read('logs/r25_gpu_budget.json');inherited=s['budget']['inherited_ledger_sha256'];assert all(sha(p)==h for p,h in inherited.items());total=sum(read(p)['used_seconds'] for p in inherited);assert book['used_seconds']==u['gpu_worker_seconds'] and book['total_used_seconds']==total+book['used_seconds'] and book['total_used_seconds']<=21600
 unmatched=0;normerror=0.;raw_mismatch=0.;czero=0;positions=0;readouts=[];nnchecks=[]
 for i,row in enumerate(rows):
  md=metadata[row['id']];historical=authority[row['id']];assert row['original_baseline_vectors_parity_error']==0 and len(row['candidates'])==len(md['request_pairs'])
  for j,c in enumerate(row['candidates']):
   ir=input_records[(row['id'],j)];assert ir['prefix_through_memory_boundary_exact'] and c['choice']==j and c['tokens']==md['choice_token_counts'][j]>0 and c['characters']==md['choice_char_lengths'][j]>0 and c['target_ids_sha256']==md['target_id_sha256'][j]==ir['target_ids_sha256'] and c['request_sha256']==__import__('r25_common').digest(md['request_pairs'][j]) and c['baseline_parity_error']==0
   for name,old in [('M2','M2_N0_token'),('M3','M3_multi_token')]:assert c['scores'][name]==historical['method_scores'][old][j] and c['metrics'][name+'_alpha']==historical['token_diagnostics'][j][old]['alpha']
   assert set(c['original_sums'])==set(historical['original_methods']) and all(v==historical['original_methods'][k]['sum_scores'][j] for k,v in c['original_sums'].items())
   assert len(c['reference_sums'])==3
   for ref,oldref in zip(c['reference_sums'],historical['prior_methods']):assert set(ref)==set(oldref) and all(v==oldref[k]['sum_scores'][j] for k,v in ref.items())
   assert set(c['scores'])==set(c['sums'])==set(NAMES) and all(math.isfinite(v) for v in list(c['scores'].values())+list(c['sums'].values())+[c['late']]);assert c['scores']['frequency']==c['scores']['M3'] and c['sums']['frequency']==c['sums']['M3']
   diag=c['metrics'];assert set(diag)==METRICS and all(len(v)==c['tokens'] for v in diag.values()) and all(math.isfinite(x) for v in diag.values() for x in v)
   for k in ['c_zero','raw_c_zero','tau_c_zero','unmatched']:assert all(type(x) is bool for x in diag[k])
   for k in ['M2_alpha','M3_alpha']:assert all(0<=x<=1 for x in diag[k])
   assert diag['c_zero']==diag['tau_c_zero'] and all(0<=x<=4 for x in diag['c_nonzero_coordinates']) and all(z==(n==0) for z,n in zip(diag['c_zero'],diag['c_nonzero_coordinates'])) and all(0<=x<=3 for x in diag['query_reference_same_value']) and all(x>=0 for k in ['main_P_norm2','matched_P_norm2'] for x in diag[k])
   mask=~np.array(diag['unmatched'],dtype=bool);unmatched+=int((~mask).sum());errors=np.abs(np.array(diag['main_P_norm2'])-diag['matched_P_norm2']);normerror=max(normerror,float(errors[mask].max(initial=0)));raw_mismatch=max(raw_mismatch,float(errors.max(initial=0)));czero+=sum(diag['c_zero']);positions+=c['tokens']
   assert len(c['reference_positions'])==3 and all(len(m)==c['tokens'] and all(type(x)is int and x>=0 for x in m) for m in c['reference_positions'])
   assert len(c['retrieval'])==4 and [r['context'] for r in c['retrieval']]==['Q','N0','N1','N2']
   for retrieval in c['retrieval']:
    assert len(retrieval['neighbor_occurrences'])==len(retrieval['minimum_two_distance_gap'])==c['tokens'] and all(type(x)is int and 0<=x<143 for x in retrieval['neighbor_occurrences']) and all(math.isfinite(x) and x>=0 for x in retrieval['minimum_two_distance_gap'])
    if 'first_numpy_check' in retrieval:
     assert i==j==0;check=retrieval['first_numpy_check'];assert check['neighbor_exact'] and math.isfinite(check['all_distance_max_error']) and check['all_distance_max_error']<=1e-12;td=np.array(check['distances_Torch']);nd=np.array(check['distances_NumPy']);assert td.shape==nd.shape==(143,) and np.isfinite(td).all() and np.isfinite(nd).all() and (td>=0).all() and (nd>=0).all() and np.max(np.abs(td-nd))<=1e-12 and int(np.argmin(td))==int(np.argmin(nd))==retrieval['neighbor_occurrences'][0];assert abs(float(np.sort(td)[1]-np.sort(td)[0])-retrieval['minimum_two_distance_gap'][0])<=1e-12;nnchecks.append(check)
   if c['readout_parity']:assert i==j==0
   readouts.extend(c['readout_parity'])
  from r7_inference import methods_from_sums
  from r7_core import baseline_vectors
  lengths=np.asarray(md['choice_char_lengths']);orig=methods_from_sums({k:[c['original_sums'][k] for c in row['candidates']] for k in row['candidates'][0]['original_sums']},lengths);prior=methods_from_sums({k:[c['reference_sums'][0][k] for c in row['candidates']] for k in row['candidates'][0]['reference_sums'][0]},lengths);baselines=baseline_vectors(orig,prior);assert all(v.tolist()==historical['baseline_scores'][k] for k,v in baselines.items())
 assert len(readouts)==4 and [x['context'] for x in readouts]==['Q','N0','N1','N2'] and all(all(v is True for k,v in x.items() if k!='context') for x in readouts) and len(nnchecks)==4 and normerror<=1e-10
 return dict(status='passed_complete_label_free_integrity',unmatched_positions=unmatched,norm_max_error=normerror,raw_norm_max_mismatch=raw_mismatch,zero_c_positions=czero,positions=positions)

def main():
 s=gate();d=read('results/r25_COMPLETE.json');u=read('logs/r25_SUPERVISOR.json');ka=read('results/r25_KEY_AUDIT.json');rows=list(jsonlines('results/r25_pilot.jsonl'));audit=integrity(s,d,u,ka,rows);save('results/r25_LABEL_FREE_INTEGRITY.json',audit)
 # ONLY after every row/candidate/numerical/schema/input/model check has passed.
 old={r['id']:r for split in ['discovery','selection'] for r in jsonlines('results/r7_'+split+'_development.jsonl')};names=NAMES+['R6','Fixed','A1'];preds={k:[] for k in names};gold=[];table=[]
 for row in rows:
  r=old[row['id']];gold.append(r['gold']);scores={k:[c['scores'][k] for c in row['candidates']] for k in NAMES}
  for k,name in [('M2','M2_N0_token'),('M3','M3_multi_token')]:assert scores[k]==r['method_scores'][name]
  for k,b in [('R6','M0_R6'),('Fixed','fixed'),('A1','A1')]:scores[k]=r['baseline_scores'][b]
  item=dict(id=row['id'],gold=r['gold'])
  for k,v in scores.items():pred=int(np.argmax(v));preds[k].append(pred);item[k+'_prediction']=pred;item[k+'_correct']=pred==r['gold']
  table.append(item)
 y=np.array(gold);counts={k:int((np.array(v)==y).sum()) for k,v in preds.items()};comparisons={k:paired(preds['primary'],v,y) for k,v in preds.items() if k!='primary'};unique=(np.array(preds['R6'])==y)&(np.array(preds['Fixed'])!=y);retained=int(((np.array(preds['primary'])==y)&unique).sum());unmatched=audit['unmatched_positions']
 gates=dict(R6_net_atleast2=comparisons['R6']['net']>=2,R6_damage0=comparisons['R6']['damages']==0,unique_present=int(unique.sum())>=1,unique_retained=retained==int(unique.sum()),no_unmatched=unmatched==0)
 for k in ['M3','M2','q_only','reference_only','memory_direction_only','raw_occurrence_shuffle','freq_bijection_matched']:gates['strictly_beats_'+k]=counts['primary']>counts[k]
 for k in ['raw_occurrence_shuffle','freq_bijection_matched']:gates['paired_net_positive_'+k]=comparisons[k]['net']>=1
 for k in ['leaveout_0','leaveout_1','leaveout_2']:
  v=np.array(preds[k]);gates[k+'_stable']=paired(v,preds['R6'],y)['net']>=0 and int(((v==y)&unique).sum())==int(unique.sum())
 strong=all(gates.values());cost=(unmatched==0 and comparisons['R6']['net']>=0 and retained==int(unique.sum()))
 save('results/r25_ANALYSIS.json',dict(status='complete24_memory_scientific_screen',n=24,counts=counts,paired=comparisons,question_results=table,R6_unique_total=int(unique.sum()),R6_unique_retained=retained,gates=gates,eligible_for_feasibility_review=strong,original_strong_screen_passed=strong,cost_review_eligible=cost,resource_policy='Pre-execution Root decision: nonpass cost eligibility is not efficacy/method pass and authorizes no more GPU',label_free_integrity=audit,protocol_sha256=d['protocol_sha256'],scores_sha256=d['score_file_sha256'],key_audit_sha256=d['key_audit_sha256'],automatic_extension_authorized=False,heldout_accessed=False,limitation='Reused development24; controls ineligible; fullchain10815seconds exceeds balance; no memory deployment/fullrun authorized'))
if __name__=='__main__':main()
