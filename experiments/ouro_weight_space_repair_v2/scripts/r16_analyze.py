"""Single predeclared pilot screen, no coefficient/threshold/control selection."""
import csv,math
import numpy as np
from r16_common import ROOT,LOCK,read,sha,digest,jsonlines,save,gate


def paired(a,b,y):
    ca=np.asarray(a)==y;cb=np.asarray(b)==y;r=int((ca&~cb).sum());d=int((cb&~ca).sum());n=r+d
    p=1. if not n else min(1.,2*sum(math.comb(n,k) for k in range(min(r,d)+1))/2**n)
    return dict(rescues=r,damages=d,net=r-d,exact_two_sided_McNemar_p=p)


def main():
    spec=gate();done=read('results/r16_pilot_COMPLETE.json');supervisor=read('logs/r16_SUPERVISOR.json')
    assert done['status']==supervisor['status']=='complete' and done['n']==64 and not done['weights_modified'] and not done['heldout_accessed']
    assert done['head_norm_before']==done['head_norm_after'] and sha(done['score_file'])==done['score_file_sha256'] and done['protocol_sha256']==sha(LOCK)
    measured=list(jsonlines(done['score_file']));assert len(measured)==len({r['id'] for r in measured})==64
    authority={r['id']:r for split in ['discovery','selection'] for r in jsonlines('results/r7_'+split+'_development.jsonl')}
    assert [r['id'] for r in measured]==[r['id'] for r in spec['cohort']]
    methods=['exact_native_KL','matched_guidance','shuffled_correction','old_Fisher','R6','fixed','A1','native4'];preds={k:[] for k in methods};gold=[];table=[];diagnostics=[];timing=[]
    for row,chosen in zip(measured,spec['cohort']):
        assert row['signature']==done['signature'] and row['cohort_entry_sha256']==digest(chosen)
        old=authority[row['id']];gold.append(old['gold']);scores={k:[c['scores'][k] for c in row['candidates']] for k in methods[:4]}
        assert scores['old_Fisher']==old['method_scores']['M2_N0_token']
        for k,b in [('R6','M0_R6'),('fixed','fixed'),('A1','A1'),('native4','native4')]:scores[k]=old['baseline_scores'][b]
        item=dict(id=row['id'],split=row['split'],gold=old['gold'])
        for k in methods:
            pred=int(np.argmax(scores[k]));preds[k].append(pred);item[k+'_prediction']=pred;item[k+'_correct']=pred==old['gold'];item[k+'_gold_margin']=scores[k][old['gold']]-max(v for j,v in enumerate(scores[k]) if j!=old['gold'])
        table.append(item)
        for c in row['candidates']:
            assert c['parity_error']==0 and all(len(v)==c['tokens'] for v in c['metrics'].values())
            diagnostics.append(c['metrics']);timing.append(c['timing'])
    y=np.asarray(gold);counts={k:int((np.asarray(v)==y).sum()) for k,v in preds.items()};comparisons={k:paired(preds['exact_native_KL'],preds[k],y) for k in methods[1:]}
    special=(np.asarray(preds['R6'])==y)&(np.asarray(preds['fixed'])!=y);retained=int(((np.asarray(preds['exact_native_KL'])==y)&special).sum())
    screens=dict(beats_R6=counts['exact_native_KL']>counts['R6'],beats_old_Fisher=counts['exact_native_KL']>counts['old_Fisher'],beats_norm_control=counts['exact_native_KL']>counts['matched_guidance'],
        beats_shuffle_control=counts['exact_native_KL']>counts['shuffled_correction'],retains_all_pilot_R6_unique_vs_Fixed=retained==int(special.sum()))
    flat=lambda k:np.asarray([x for d in diagnostics for x in d[k]])
    a=flat('alpha');old=flat('old_Fisher_alpha');width=flat('alpha_interval_width');bound=flat('numerical_bound');ideal=flat('objective_ideal');actual=flat('objective_FP32');rounding=flat('max_logit_rounding')
    assert np.all(width<=2**-32) and np.all(ideal<=bound) and np.all(ideal<=flat('old_Fisher_objective_ideal')+bound) and np.all(actual<=ideal+2*rounding+bound)
    assert np.max(np.abs(flat('matched_guidance_norm2')-flat('residual_norm2')))<=1e-10 and np.max(np.abs(flat('shuffled_correction_norm2')-flat('correction_norm2')))<=1e-10
    with (ROOT/'results/r16_question_results.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    result=dict(status='complete_single_fixed_pilot',n=64,protocol_sha256=sha(LOCK),counts=counts,paired= comparisons,R6_unique_vs_Fixed=dict(total=int(special.sum()),retained=retained),screens=screens,
        eligible_for_feasibility_review=all(screens.values()),automatic_extension_authorized=False,heldout_accessed=False,external_benchmarks_started=False,
        numerical=dict(positions=len(a),alpha_zero=int((a==0).sum()),alpha_one=int((a==1).sum()),interior=int(((a>0)&(a<1)).sum()),new_active_old_zero=int(((a>0)&(old==0)).sum()),
            mean_absolute_alpha_change=float(np.abs(a-old).mean()),max_absolute_alpha_change=float(np.abs(a-old).max()),rho_greater_than_one=int((flat('rho')>1).sum()),max_interval_width=float(width.max()),
            mean_KL_objective_gain_over_old_Fisher=float((flat('old_Fisher_objective_ideal')-ideal).mean()),max_root_residual_interior=float(np.abs(flat('root_derivative')[(a>0)&(a<1)]).max(initial=0))),
        timing_sums={k:sum(t[k] for t in timing) for k in timing[0]},model_load_seconds=done['model_load_seconds'],worker_elapsed_seconds=done['elapsed_seconds'],
        limitation='Post-evaluation development audit, one uniform64 sample from previously reused576. J1 only, reference stability untested; KL optimum does not guarantee correctness. Cached main, no standalone deployment timing.')
    save('results/r16_PILOT_ANALYSIS.json',result);print(result['status'],counts,screens,flush=True)


if __name__=='__main__':main()
