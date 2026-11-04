"""Analyze only complete, non-overrun R20 cohort; no partial effects."""
import numpy as np
from r20_common import read,sha,save,gate,jsonlines
from r19_analyze import paired

def main():
    spec=gate();done=read('results/r20_pilot_COMPLETE.json');sup=read('logs/r20_SUPERVISOR.json')
    assert done['status']==sup['status']=='complete' and done['n']==24 and not sup['overrun'] and not sup.get('timed_out',False)
    assert done['protocol_sha256']==sha('configs/r20_VALUE_ABLATION_PILOT_LOCK.json') and done['score_file_sha256']==sha('results/r20_pilot.jsonl')
    assert done['all_parameter_hashes_before']==done['all_parameter_hashes_after'] and not done['weights_modified']
    rows=list(jsonlines('results/r20_pilot.jsonl'));assert [r['id'] for r in rows]==[c['id'] for c in spec['cohort']]
    first=rows[0]['candidates'][0];assert all(first['first_candidate_exact_checks'].values()) and len(first['actual_forward_audit'])==5
    old={r['id']:r for s in ['discovery','selection'] for r in jsonlines('results/r7_'+s+'_development.jsonl')}
    names=['primary','demo','raw_demo_matched','shuffle','old_Fisher','R6','Fixed','A1'];preds={k:[] for k in names};gold=[];table=[];normerror=0.;raw_mismatch=0.;unmatched=0
    for row in rows:
        r=old[row['id']];gold.append(r['gold']);scores={k:[x['scores'][k] for x in row['candidates']] for k in names[:5]}
        assert scores['old_Fisher']==r['method_scores']['M2_N0_token']
        for k,b in [('R6','M0_R6'),('Fixed','fixed'),('A1','A1')]:scores[k]=r['baseline_scores'][b]
        item=dict(id=row['id'],gold=r['gold'])
        for k,v in scores.items():pred=int(np.argmax(v));preds[k].append(pred);item[k+'_prediction']=pred;item[k+'_correct']=pred==r['gold']
        table.append(item)
        for c in row['candidates']:
            assert c['parity_error']==0 and all(len(v)==c['tokens'] for v in c['metrics'].values())
            d=c['metrics'];unmatched+=sum(d['unmatched'])
            raw=np.abs(np.array(d['matched_norm2'])-d['main_norm2']);raw_mismatch=max(raw_mismatch,float(raw.max()));matched=~np.array(d['unmatched'],dtype=bool)
            normerror=max(normerror,float(raw[matched].max(initial=0)),float(np.max(np.abs(np.array(d['shuffle_norm2'])-d['main_norm2']))))
    assert normerror<=1e-10
    y=np.array(gold);counts={k:int((np.array(v)==y).sum()) for k,v in preds.items()};comparisons={k:paired(preds['primary'],preds[k],y) for k in names[1:]}
    unique=(np.array(preds['R6'])==y)&(np.array(preds['Fixed'])!=y);retained=int(((np.array(preds['primary'])==y)&unique).sum())
    checks=dict(R6_net_at_least_two=comparisons['R6']['net']>=2,R6_no_damage=comparisons['R6']['damages']==0,R6_unique_present=int(unique.sum())>=1,R6_unique_retained=retained==int(unique.sum()),controls_matched=unmatched==0)
    checks.update({'strictly_exceeds_'+k:counts['primary']>counts[k] for k in ['old_Fisher','demo','raw_demo_matched','shuffle']})
    checks.update({'paired_net_positive_'+k:comparisons[k]['net']>=1 for k in ['demo','raw_demo_matched']})
    save('results/r20_PILOT_ANALYSIS.json',dict(status='complete_single_pilot',n=24,counts=counts,paired=comparisons,question_results=table,R6_unique_total=int(unique.sum()),R6_unique_retained=retained,norm_max_error=normerror,raw_C_max_mismatch=raw_mismatch,protocol_sha256=done['protocol_sha256'],score_file_sha256=done['score_file_sha256'],source_sha256=spec['source_sha256'],model_identity=spec['model_identity'],unmatched_positions=unmatched,screens=checks,eligible_for_feasibility_review=all(checks.values()),automatic_extension_authorized=False,heldout_accessed=False,limitation='Repeated development24, no independent generalization or significance claim; controls cannot be promoted.'))
if __name__=='__main__':main()


