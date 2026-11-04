"""Post-screen development failure accounting, not a method or causal semantic claim."""
import csv,math
from r19_common import ROOT,read,jsonlines,save,gate


def main():
    gate();analysis=read('results/r19_PILOT_ANALYSIS.json');assert analysis['status']=='complete_single_last_loop_pilot'
    rows=list(jsonlines('results/r19_pilot.jsonl'));base={r['id']:r for s in ['discovery','selection'] for r in jsonlines('results/r7_'+s+'_development.jsonl')}
    details=[];cases=[];changed=0
    for row in rows:
        old=base[row['id']];by={c['choice']:c for c in row['candidates']};y=old['gold'];new=max(by,key=lambda j:by[j]['scores']['paired_loop_protection']);previous=max(by,key=lambda j:by[j]['scores']['old_Fisher'])
        changed+=new!=previous;r6=old['predictions']['M0_R6']
        if (new==y)==(r6==y):continue
        rival=max((j for j in by if j!=y),key=lambda j:by[j]['scores']['paired_loop_protection']);kind='rescue' if new==y else 'damage'
        def component(which,key):return math.fsum(by[which]['metrics'][key])/by[which]['characters']
        target=component(y,'paired_loop_protection_target_shift')-component(rival,'paired_loop_protection_target_shift')
        normalizer=component(y,'paired_loop_protection_normalizer_change')-component(rival,'paired_loop_protection_normalizer_change')
        predmargin=by[y]['scores']['paired_loop_protection']-by[rival]['scores']['paired_loop_protection'];a1margin=old['baseline_scores']['A1'][y]-old['baseline_scores']['A1'][rival]
        source=next(q for s in ['discovery','selection'] for q in jsonlines('data/slcd_'+s+'.jsonl') if q['id']==row['id'])
        case=dict(id=row['id'],transition=kind,gold=y,R6_prediction=r6,new_prediction=new,old_Fisher_prediction=previous,post_repair_rival=rival,
            original_question=source['question'],original_choices=source['choices'],
            R6_pair_margin=old['baseline_scores']['M0_R6'][y]-old['baseline_scores']['M0_R6'][rival],A1_pair_margin=a1margin,new_pair_margin=predmargin,
            target_shift_pair_change=target,normalizer_pair_change=normalizer,LP_pair_change=target-normalizer,reconstruction_error=predmargin-a1margin-(target-normalizer),
            convention='Positive target shift improves gold margin; positive normalizer change opposes gold. Pair and rival fixed after observed repair; gauge-dependent score accounting, not semantic causal proof.')
        cases.append(case)
        for c in row['candidates']:
            m=c['metrics']
            for t in range(c['tokens']):details.append(dict(id=row['id'],transition=kind,choice=c['choice'],is_gold=c['choice']==y,is_rival=c['choice']==rival,position=t,characters=c['characters'],
                **{k:m[k][t] for k in ['alpha','old_Fisher_alpha','rho','protection_b','s_norm2','e_norm2','residual_prior_norm2','axis_outside_ge_span_fraction','orthogonality_pre','orthogonality_rounded','paired_loop_protection_target_shift','paired_loop_protection_normalizer_change','paired_loop_protection_LP_change']}))
    if details:
        with (ROOT/'results/r19_failure_prefixes.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    assert len([c for c in cases if c['transition']=='damage'])==analysis['paired']['R6']['damages']
    assert len([c for c in cases if c['transition']=='rescue'])==analysis['paired']['R6']['rescues']
    result=dict(status='complete_post_screen_failure_accounting',n=len(rows),new_vs_old_Fisher_prediction_disagreements=changed,cases=cases,
        prefix_rows=len(details),formula_changed=False,heldout_accessed=False,scope='Fixed post-repair gold/rival accounting only. No threshold, dispatch, new candidate or independent causal explanation is inferred.')
    save('results/r19_FAILURE_CASES.json',result);print('CASE_AUDIT_COMPLETE',changed,len(cases),len(details))


if __name__=='__main__':main()
