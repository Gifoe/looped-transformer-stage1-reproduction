"""Post-run audit only; no state loading or new association tests."""
import json,math
from r22_common import ROOT,sha,read,save

def main():
 s=read('configs/r22_TRAJECTORY_LOCK.json');d=read('results/r22_FEATURES_COMPLETE.json');u=read('logs/r22_SUPERVISOR.json');a=read('results/r22_ANALYSIS.json');m=read('configs/r22_CHALLENGE_MANIFEST.json')['rows'];f=[json.loads(x) for x in (ROOT/'results/r22_features.jsonl').read_text().splitlines()]
 assert d['status']=='complete_gold_free_features' and d['n']==len(f)==576 and u['status']=='complete' and u['exit_code']==0 and not u['overrun']
 assert d['features_sha256']==sha('results/r22_features.jsonl') and d['protocol_sha256']==sha('configs/r22_TRAJECTORY_LOCK.json')
 assert [x['id'] for x in f]==[x['id'] for x in m] and all(sha('scripts/'+p)==h for p,h in s['source_sha256'].items())
 for row,feature in zip(m,f):
  assert feature['all_candidates_prefix_ids_exact'] and feature['first_position_maxdiff_by_depth']==[0]*6 and len(feature['candidates'])==row['state']['candidate_count'] and feature['state_sha256']==row['state']['sha256']
  for j,c in enumerate(feature['candidates']):
   assert c['candidate']==j and c['tokens']==row['state']['choice_token_counts'][j] and all(math.isfinite(c[k]) for k in ['V','C_energy','C_margin','C_log_tokens','C_neg_log_tokens'])
   assert 0<=c['V']<=1 and c['degenerate']>=0
  for j in row['challengers']:
   b=row['R6_winner']
   if feature['candidates'][j]['tokens']==feature['candidates'][b]['tokens']==1:assert feature['candidates'][j]['V']==feature['candidates'][b]['V']
 save('results/r22_EXECUTION_VERIFIED.json',dict(status='complete576_audits_passed_specific_signal_gate_failed',features_sha256=d['features_sha256'],protocol_sha256=d['protocol_sha256'],full576_prefix_and_firstsixstates_exact=True,all54_rescue_IDs_covered=a['all54_R21_rescue_questions_present'],first_NumPy_Torch_max_error=d['first_real_checks'][0]['first_NumPy_Torch_max_error'],CPU_supervised_seconds=u['CPU_wall_seconds'],GPU_worker_seconds=0,model_loaded=False,degenerate_positions=a['degenerate_positions'],positions=a['total_positions'],screens=a['screens'],worth_further_research=a['worth_further_research'],no_additional_feature_or_group_tests=True))
 print('R22_EXECUTION_AUDIT_PASS')
if __name__=='__main__':main()
