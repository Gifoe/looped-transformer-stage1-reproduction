"""Label analysis only after all576 audited gold-free features completed."""
import json
import numpy as np
from r22_common import ROOT,read,sha,save,gate
from r22_statistics import SIGNALS,evaluate

def main():
 spec=gate();done=read('results/r22_FEATURES_COMPLETE.json');sup=read('logs/r22_SUPERVISOR.json');assert done['status']=='complete_gold_free_features' and done['n']==576 and sup['status']=='complete' and not sup['overrun'] and not sup.get('timed_out',False)
 assert done['features_sha256']==sha('results/r22_features.jsonl') and done['protocol_sha256']==sha('configs/r22_TRAJECTORY_LOCK.json')
 manifest=read('configs/r22_CHALLENGE_MANIFEST.json')['rows'];features=[json.loads(x) for x in (ROOT/'results/r22_features.jsonl').read_text().splitlines()];assert [r['id'] for r in features]==[r['id'] for r in manifest]
 labels={r['id']:r['gold'] for split in ['discovery','selection'] for r in [json.loads(x) for x in (ROOT/('results/r7_'+split+'_development.jsonl')).read_text().splitlines()]}
 pairs=[];rescueids=set();degenerate=0;positions=0
 for row,f in zip(manifest,features):
  assert f['all_candidates_prefix_ids_exact'] and all(v==0 for v in f['first_position_maxdiff_by_depth'])
  c=f['candidates'];b=row['R6_winner'];gold=labels[row['id']]
  for value in c:degenerate+=value['degenerate'];positions+=value['tokens']
  for a in row['challengers']:
   category='damage' if b==gold else 'rescue' if a==gold else 'background'
   if category=='rescue':rescueids.add(row['id'])
   pairs.append(dict(id=row['id'],split=row['split'],category=category,delta=[c[a][k]-c[b][k] for k in SIGNALS],equal_tokens=c[a]['tokens']==c[b]['tokens'],both_single=c[a]['tokens']==c[b]['tokens']==1))
 r21=[json.loads(x) for x in (ROOT/'results/r21_oracle.jsonl').read_text().splitlines()];expected_rescues={r['id'] for r in r21 if not r['R6_correct'] and r['discrete_reachable']};assert rescueids==expected_rescues and len(rescueids)==54
 results={};coverage={}
 for split in ['discovery','selection','combined']:
  ps=[p for p in pairs if split=='combined' or p['split']==split];groups=[]
  for category in ['rescue','damage']:
   ids=sorted({p['id'] for p in ps if p['category']==category});groups.append([np.array([p['delta'] for p in ps if p['id']==qid and p['category']==category]) for qid in ids])
  results[split]=evaluate(*groups)
  coverage[split]={}
  for group in ['all','equal_tokens','both_single']:
   subset=[p for p in ps if group=='all' or p[group]];coverage[split][group]={k:dict(questions=len({p['id'] for p in subset if p['category']==k}),pairs=sum(p['category']==k for p in subset)) for k in ['rescue','damage','background']}
 adequate=all(results[s]['rescue_questions']>=10 and results[s]['damage_questions']>=10 for s in ['discovery','selection']);combined=results['combined']
 screens=dict(power=adequate,split_AUC=all(results[s].get('AUC',{}).get('V',-1)>=.65 for s in ['discovery','selection']),combined_CI_lower_above_half=combined.get('CI95',{}).get('V',[-1])[0]>.5,all_four_control_differences_CI_positive=all(combined.get('paired_difference',{}).get(k,{'CI95':[-1]})['CI95'][0]>0 for k in SIGNALS[1:]))
 save('results/r22_ANALYSIS.json',dict(status='complete_label_association',n=576,results=results,coverage=coverage,all54_R21_rescue_questions_present=True,degenerate_positions=degenerate,total_positions=positions,degenerate_fraction=degenerate/positions,screens=screens,worth_further_research=all(screens.values()),interpretation_status='underpowered_inconclusive' if not adequate else 'association_gate_passed' if all(screens.values()) else 'specific_hypothesis_gate_failed',protocol_sha256=done['protocol_sha256'],features_sha256=done['features_sha256'],controls_ineligible=True,scope='Repeated-development association, not causal/deployable; coverage subgroups descriptive only'))
 print('R22_ANALYSIS_COMPLETE')
if __name__=='__main__':main()
