"""All-condition postrun summaries from fully unsealed frozen outputs; CPU only."""
from pathlib import Path
import csv,json,hashlib,collections
O=Path(__file__).resolve().parent
read=lambda p:json.loads((O/p).read_text(encoding='utf8'))
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf8',newline='')))
def main():
 assert read('ANALYSIS.json')['status']=='complete24_all_conditions_numeric_before_labels'
 native=rows('native_predictions.csv');perm=rows('permutation_transitions.csv');loop=rows('native_loop_transitions.csv');side=read('CONSTRUCTOR_WITH_LABELS.json');first4=side['selection']['bounded_B5_B6_ids'];overview=[]
 for c in sorted({r['condition'] for r in native}):
  for k in range(1,5):
   rr=[r for r in native if r['condition']==c and r['K']==str(k)];overview.append(dict(condition=c,K=k,n=len(rr),raw_target_match=sum(r['target_match']=='True' for r in rr),char_normalized_target_match=sum(int(r['char_normalized_prediction'])==int(r['expected_display_index']) for r in rr)))
 b5=[]
 for c,label in [('P0','B5_L_letter'),('B5_L_full','B5_L_full'),('B5_F_letter','B5_F_letter'),('B5_F_full','B5_F_full')]:
  for k in range(1,5):
   rr=[r for r in native if r['condition']==c and r['K']==str(k) and r['id'] in first4];assert len(rr)==4;b5.append(dict(condition=label,K=k,n=4,raw_target_match=sum(r['target_match']=='True' for r in rr),char_normalized_target_match=sum(int(r['char_normalized_prediction'])==int(r['expected_display_index']) for r in rr)))
 binding=[]
 for c in ['P1','P2','P3']:
  for k in range(1,5):
   rr=[r for r in perm if r['condition']==c and r['K']==str(k)];m=[r for r in rr if r['P0_predicted_content_moved']=='True'];binding.append(dict(condition=c,K=k,n=24,content_equivariant=sum(r['content_equivariant']=='True' for r in rr),display_letter_invariant=sum(r['display_letter_invariant']=='True' for r in rr),moved_denominator=len(m),moved_content_equivariant=sum(r['content_equivariant']=='True' for r in m),moved_display_letter_invariant=sum(r['display_letter_invariant']=='True' for r in m)))
 transitions=[]
 for c in sorted({r['condition'] for r in loop}):
  for k in [1,2,3]:
   rr=[r for r in loop if r['condition']==c and r['K_from']==str(k) and r['readout']=='raw'];transitions.append(dict(condition=c,K_from=k,K_to=k+1,n=len(rr),rescue=sum(r['rescue']=='True' for r in rr),damage=sum(r['damage']=='True' for r in rr),changed=sum(r['content_changed']=='True' for r in rr)))
 for name,rr in [('overview_by_depth.csv',overview),('B5_same4_matrix.csv',b5),('binding_overview.csv',binding),('loop_transition_overview.csv',transitions)]:
  with (O/name).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 budget=read('gpu_budget.json');timing=read('TIMINGS.json');components={name:sum(r[name] for r in timing) for name in ['forward_and_four_heads_seconds','scoring_diagnostics_seconds','vector_export_seconds']};worker=read('COMPLETE.json')['worker_elapsed_seconds'];components['worker_unclassified_startup_tokenizer_hash_final_IO_residual_seconds']=worker-sum(components.values());components['supervisor_launch_exit_reap_overhead_seconds']=budget['used_seconds']-worker
 inherited=read('EXECUTION_LOCK.json')['inherited_ledgers'];refs=[dict(path=n,sha256=h,used_seconds=json.loads((O.parent/n).read_text(encoding='utf8'))['used_seconds']) for n,h in inherited.items()];assert sum(x['used_seconds'] for x in refs)+budget['used_seconds']==budget['total_used_seconds']
 ledger=dict(inherited_ledgers=refs,inherited_count=20,inherited_used_seconds=budget['inherited_used_seconds'],new_ledger='gpu_budget.json',new_ledger_sha256=hashlib.sha256((O/'gpu_budget.json').read_bytes()).hexdigest(),new_GPU_worker_seconds=budget['used_seconds'],total_ledger_count=21,total_used_seconds=budget['total_used_seconds'],cap_seconds=21600,remaining_seconds=budget['remaining_seconds'],components=components,scope='Components are not added again to actualsupervisor wall; CPU-only audit/archival isGPU0; sharedGB10 diagnostic pipeline not deployment timing.')
 (O/'compute_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n',encoding='utf8');(O/'POSTRUN_SUMMARY.json').write_text(json.dumps(dict(overview=overview,B5_same4=b5,binding=binding,transitions=transitions,scope='All fixed questions/conditions/depths, no new labels or model work.'),indent=2)+'\n',encoding='utf8');print(json.dumps(dict(P0=[r for r in overview if r['condition']=='P0'],P0trans=[r for r in transitions if r['condition']=='P0'],B5=b5),indent=2))
if __name__=='__main__':main()
