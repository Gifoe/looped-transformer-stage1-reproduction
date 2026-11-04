"""One finite paired-last-loop ablation; reference-only control cannot win."""
import csv,math
import numpy as np
from r19_common import ROOT,LOCK,read,sha,digest,jsonlines,save,gate
METHODS=['paired_loop_protection','reference_only_axis','matched_guidance','shuffled_correction','shuffled_axis','old_Fisher','R6','fixed','A1','native4']


def paired(a,b,y):
    ca=np.asarray(a)==y;cb=np.asarray(b)==y;r=int((ca&~cb).sum());d=int((cb&~ca).sum());n=r+d
    return dict(rescues=r,damages=d,net=r-d,exact_two_sided_McNemar_p=1. if n==0 else min(1.,2*sum(math.comb(n,k) for k in range(min(r,d)+1))/2**n))


def main():
    spec=gate();done=read('results/r19_pilot_COMPLETE.json');sup=read('logs/r19_SUPERVISOR.json')
    assert done['status']==sup['status']=='complete' and done['n']==64 and not done['weights_modified'] and not done['heldout_accessed']
    assert done['protocol_sha256']==sha(LOCK) and sha(done['score_file'])==done['score_file_sha256'] and done['head_norm_before']==done['head_norm_after']
    records=list(jsonlines(done['score_file']));assert [r['id'] for r in records]==[r['id'] for r in spec['cohort']]
    authority={r['id']:r for s in ['discovery','selection'] for r in jsonlines('results/r7_'+s+'_development.jsonl')};preds={k:[] for k in METHODS};gold=[];table=[];diags=[];timing=[];checks=[]
    for row,c in zip(records,spec['cohort']):
        assert row['signature']==done['signature']==digest(dict(protocol_sha256=sha(LOCK),cohort=spec['cohort'])) and row['cohort_entry_sha256']==digest(c)
        old=authority[row['id']];gold.append(old['gold']);scores={k:[x['scores'][k] for x in row['candidates']] for k in METHODS[:6]};assert scores['old_Fisher']==old['method_scores']['M2_N0_token']
        for k,b in [('R6','M0_R6'),('fixed','fixed'),('A1','A1'),('native4','native4')]:scores[k]=old['baseline_scores'][b]
        item=dict(id=row['id'],split=row['split'],gold=old['gold'])
        for k,v in scores.items():
            pred=int(np.argmax(v));preds[k].append(pred);item[k+'_prediction']=pred;item[k+'_correct']=pred==old['gold'];item[k+'_gold_margin']=v[old['gold']]-max(x for j,x in enumerate(v) if j!=old['gold'])
        table.append(item)
        for x in row['candidates']:
            assert x['parity_error']==0 and all(len(v)==x['tokens'] for v in x['metrics'].values());diags.append(x['metrics']);timing.append(x['timing'])
            if x['independent_depth3_check'] is not None:checks.append(x['independent_depth3_check'])
    assert len(checks)==1 and all(checks[0].values())
    y=np.asarray(gold);counts={k:int((np.asarray(v)==y).sum()) for k,v in preds.items()};comparisons={k:paired(preds['paired_loop_protection'],preds[k],y) for k in METHODS[1:]}
    special=(np.asarray(preds['R6'])==y)&(np.asarray(preds['fixed'])!=y);retained=int(((np.asarray(preds['paired_loop_protection'])==y)&special).sum())
    screens={'beats_'+k:counts['paired_loop_protection']>counts[k] for k in ['R6','old_Fisher','reference_only_axis','matched_guidance','shuffled_correction','shuffled_axis']};screens['retains_all_pilot_R6_unique_vs_Fixed']=retained==int(special.sum())
    flat=lambda k:np.asarray([v for d in diags for v in d[k]]);a=flat('alpha');old=flat('old_Fisher_alpha');defined=flat('orthogonality_defined').astype(bool)
    assert np.all((a>=0)&(a<=1)) and np.all(np.abs(flat('orthogonality_pre')[defined])<=flat('orthogonality_arithmetic_bound')[defined]) and np.all(np.abs(flat('orthogonality_rounded'))<=flat('orthogonality_bound'))
    errors={}
    for k,v in [('matched_guidance_norm2','residual_norm2'),('shuffled_correction_norm2','correction_norm2')]:errors[k]=float(np.abs(flat(k)-flat(v)).max());assert errors[k]<=1e-10
    with (ROOT/'results/r19_question_results.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    out=dict(status='complete_single_last_loop_pilot',n=64,protocol_sha256=sha(LOCK),counts=counts,paired=comparisons,R6_unique_vs_Fixed=dict(total=int(special.sum()),retained=retained),screens=screens,
        eligible_for_feasibility_review=all(screens.values()),automatic_extension_authorized=False,heldout_accessed=False,external_benchmarks_started=False,
        independent_depth3_checks=checks[0],primary_vs_reference_only_prediction_disagreements=int((np.asarray(preds['paired_loop_protection'])!=np.asarray(preds['reference_only_axis'])).sum()),
        numerical=dict(positions=len(a),alpha_zero=int((a==0).sum()),alpha_one=int((a==1).sum()),mean_absolute_alpha_change=float(np.abs(a-old).mean()),max_absolute_alpha_change=float(np.abs(a-old).max()),axis_unresolved=int(flat('s_unresolved').sum()),
            pre_FP32_norm_errors=errors,orthogonality_defined_positions=int(defined.sum()),max_orthogonality_pre=float(np.abs(flat('orthogonality_pre')[defined]).max(initial=0)),max_orthogonality_rounded=float(np.abs(flat('orthogonality_rounded')).max()),
            axis_outside_ge_span_fraction_quantiles=np.quantile(flat('axis_outside_ge_span_fraction'),[0,.25,.5,.75,1]).tolist(),
            interpretation='Outside-span fraction is descriptive under numerically resolved native Fisher geometry. Axis sign cannot determine answer correctness. Orthogonal FP64 logit components do not ensure probability/answer preservation.'),
        timing_sums={k:sum(t[k] for t in timing) for k in timing[0]},model_load_seconds=done['model_load_seconds'],worker_elapsed_seconds=done['elapsed_seconds'],
        limitation='Reused development64, fixedN0, no independent validation or J3 stability claim. If reference-only matches/betters primary, no evidence that the added real-question loop3 information improves this repair.')
    save('results/r19_PILOT_ANALYSIS.json',out);print(out['status'],counts,screens,flush=True)


if __name__=='__main__':main()
