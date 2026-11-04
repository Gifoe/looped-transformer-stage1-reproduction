"""R23 analysis only complete24 clean supervisor all-parameter identity."""
import numpy as np
from r23_common import ROOT,LOCK,read,sha,save,gate,jsonlines
from r19_analyze import paired

def main():
 spec=gate();d=read('results/r23_COMPLETE.json');u=read('logs/r23_SUPERVISOR.json');assert d['status']==u['status']=='complete' and d['n']==24 and not u['overrun'] and not u.get('timed_out',False) and d['all_parameter_hashes_before']==d['all_parameter_hashes_after']
 assert d['protocol_sha256']==sha(LOCK) and d['score_file_sha256']==sha('results/r23_pilot.jsonl')
 rows=list(jsonlines('results/r23_pilot.jsonl'));assert [r['id'] for r in rows]==[x['id'] for x in spec['cohort']];old={r['id']:r for s in ['discovery','selection'] for r in jsonlines('results/r7_'+s+'_development.jsonl')}
 names=['primary','M3','M2','q_only','reference_only','shuffle','probability_contrast','leaveout_0','leaveout_1','leaveout_2','R6','Fixed','A1'];preds={k:[] for k in names};gold=[];table=[];unmatched=0;normerror=0.;readout=[]
 for row in rows:
  assert row['original_baseline_vectors_parity_error']==0
  r=old[row['id']];gold.append(r['gold']);scores={k:[x['scores'][k] for x in row['candidates']] for k in names[:10]}
  for k,name in [('M2','M2_N0_token'),('M3','M3_multi_token')]:assert scores[k]==r['method_scores'][name]
  for k,b in [('R6','M0_R6'),('Fixed','fixed'),('A1','A1')]:scores[k]=r['baseline_scores'][b]
  item=dict(id=row['id'],gold=r['gold'])
  for k,v in scores.items():pred=int(np.argmax(v));preds[k].append(pred);item[k+'_prediction']=pred;item[k+'_correct']=pred==r['gold']
  table.append(item)
  for c in row['candidates']:
   assert c['baseline_parity_error']==c['native2_native3_parity_error']==0;readout+=c['readout_checks'];dd=c['metrics']
   for n,bad in [('shuffle_P_norm2','shuffle_unmatched'),('probability_P_norm2','probability_unmatched')]:
    mask=~np.array(dd[bad],dtype=bool);unmatched+=int((~mask).sum());normerror=max(normerror,float(np.abs(np.array(dd[n])-dd['c_P_norm2'])[mask].max(initial=0)))
 assert len(readout)==8 and all(r['fullV_exact'] for r in readout) and normerror<=1e-10
 y=np.array(gold);counts={k:int((np.array(v)==y).sum()) for k,v in preds.items()};comparisons={k:paired(preds['primary'],v,y) for k,v in preds.items() if k!='primary'};unique=(np.array(preds['R6'])==y)&(np.array(preds['Fixed'])!=y);retained=int(((np.array(preds['primary'])==y)&unique).sum())
 gates=dict(R6_net_at_least2=comparisons['R6']['net']>=2,R6_damage_zero=comparisons['R6']['damages']==0,unique_present=int(unique.sum())>=1,unique_retained=retained==int(unique.sum()),no_unmatched=unmatched==0,probability_contrast_paired_net_positive=comparisons['probability_contrast']['net']>=1)
 gates.update({'strictly_beats_'+k:counts['primary']>counts[k] for k in ['M3','M2','q_only','reference_only','shuffle','probability_contrast']})
 for k in ['leaveout_0','leaveout_1','leaveout_2']:
  v=np.array(preds[k]);p=paired(v,preds['R6'],y);gates[k+'_stable']=p['net']>=0 and int(((v==y)&unique).sum())==int(unique.sum())
 save('results/r23_ANALYSIS.json',dict(status='complete24_scientific_screen',n=24,counts=counts,paired=comparisons,question_results=table,R6_unique_total=int(unique.sum()),R6_unique_retained=retained,gates=gates,eligible_for_feasibility_review=all(gates.values()),unmatched_positions=unmatched,norm_max_error=normerror,independent_native2_3_checks=readout,protocol_sha256=d['protocol_sha256'],scores_sha256=d['score_file_sha256'],automatic_extension_authorized=False,heldout_accessed=False,limitation='Reused development24 exploratory pilot; controls ineligible; fullchain exceeds remaining budget'))
if __name__=='__main__':main()
