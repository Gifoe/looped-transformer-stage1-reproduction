"""One fixed metric ablation screen and descriptive rank diagnostics."""
import csv,math
import numpy as np
from r18_common import ROOT,LOCK,read,sha,digest,jsonlines,save,gate

METHODS=['uniform_projection','matched_guidance_U','matched_guidance_P','shuffled_correction_U','shuffled_correction_P','old_Fisher','R6','fixed','A1','native4']


def paired(a,b,y):
    ca=np.asarray(a)==y;cb=np.asarray(b)==y;r=int((ca&~cb).sum());d=int((cb&~ca).sum());n=r+d
    p=1. if n==0 else min(1.,2*sum(math.comb(n,k) for k in range(min(r,d)+1))/2**n)
    return dict(rescues=r,damages=d,net=r-d,exact_two_sided_McNemar_p=p)


def main():
    spec=gate();done=read('results/r18_pilot_COMPLETE.json');supervisor=read('logs/r18_SUPERVISOR.json')
    assert done['status']==supervisor['status']=='complete' and done['n']==64 and not done['weights_modified'] and not done['heldout_accessed']
    assert done['head_norm_before']==done['head_norm_after'] and sha(done['score_file'])==done['score_file_sha256'] and done['protocol_sha256']==sha(LOCK)
    records=list(jsonlines(done['score_file']));assert [r['id'] for r in records]==[r['id'] for r in spec['cohort']]
    authority={r['id']:r for s in ['discovery','selection'] for r in jsonlines('results/r7_'+s+'_development.jsonl')};preds={k:[] for k in METHODS};gold=[];table=[];diags=[];timing=[]
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
    y=np.asarray(gold);counts={k:int((np.asarray(v)==y).sum()) for k,v in preds.items()};comparisons={k:paired(preds['uniform_projection'],preds[k],y) for k in METHODS[1:]}
    special=(np.asarray(preds['R6'])==y)&(np.asarray(preds['fixed'])!=y);retained=int(((np.asarray(preds['uniform_projection'])==y)&special).sum())
    screens={'beats_'+k:counts['uniform_projection']>counts[k] for k in ['R6','old_Fisher','matched_guidance_U','matched_guidance_P','shuffled_correction_U','shuffled_correction_P']};screens['retains_all_pilot_R6_unique_vs_Fixed']=retained==int(special.sum())
    flat=lambda k:np.asarray([v for d in diags for v in d[k]]);a=flat('alpha');old=flat('old_Fisher_alpha');valid=flat('Fisher_rank_defined').astype(bool);rank=flat('Fisher_effective_rank')[valid]
    assert np.all((a>=0)&(a<=1))
    normerrors={}
    for k,v in [('matched_U_norm2','residual_U_norm2'),('matched_P_norm2','residual_P_norm2'),('shuffle_U_norm2','correction_U_norm2'),('shuffle_P_norm2','correction_P_norm2')]:
        normerrors[k]=float(np.abs(flat(k)-flat(v)).max());assert normerrors[k]<=1e-10
    quant=lambda x:np.quantile(x,[0,.25,.5,.75,1]).tolist() if len(x) else None
    align={}
    for name in ['uniform','native','uniform_shuffle','native_shuffle']:
        ok=flat(name+'_alignment_defined').astype(bool);v=flat(name+'_alignment')[ok];align[name]=dict(defined_positions=int(ok.sum()),mean_cosine=float(v.mean()) if len(v) else None,mean_squared_cosine=float((v*v).mean()) if len(v) else None)
    with (ROOT/'results/r18_question_results.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    result=dict(status='complete_single_fixed_metric_pilot',n=64,protocol_sha256=sha(LOCK),counts=counts,paired=comparisons,R6_unique_vs_Fixed=dict(total=int(special.sum()),retained=retained),screens=screens,
        eligible_for_feasibility_review=all(screens.values()),automatic_extension_authorized=False,heldout_accessed=False,external_benchmarks_started=False,
        primary_vs_old_Fisher_prediction_disagreements=int((np.asarray(preds['uniform_projection'])!=np.asarray(preds['old_Fisher'])).sum()),
        numerical=dict(positions=len(a),alpha_zero=int((a==0).sum()),alpha_one=int((a==1).sum()),mean_absolute_alpha_change=float(np.abs(a-old).mean()),max_absolute_alpha_change=float(np.abs(a-old).max()),
            pre_FP32_control_norm_errors=normerrors,rho_P_greater_than_one=int((flat('rho_P')>1).sum())),
        geometry_diagnostic=dict(scope='Pooled prefix summaries are descriptive, not independent samples. Original g/e alignment, not R17 cue-response remeasurement.',
            effective_Fisher_rank_defined=int(valid.sum()),effective_Fisher_rank_undefined=int((~valid).sum()),effective_Fisher_rank_quantiles=quant(rank),probability_IPR_quantiles=quant(flat('native_probability_IPR')),pmax_quantiles=quant(flat('pmax')),alignment=align,
            correction_energy_means={k:float(flat(k).mean()) for k in ['correction_U_norm2','correction_P_norm2','old_Fisher_correction_U_norm2','old_Fisher_correction_P_norm2']},
            interpretation='Low Fisher rank is not a computation error or proof of irrelevant reference; fixed vocabulary shuffle is not an isotropic null. Uniform tails may dominate and do not supply semantic ground truth.'),
        timing_sums={k:sum(t[k] for t in timing) for k in timing[0]},model_load_seconds=done['model_load_seconds'],worker_elapsed_seconds=done['elapsed_seconds'],
        limitation='One reused-development64 pilot, N0 only, no reference-stability claim and no independent validation. Uniform full-vocabulary prior-direction metric differs from old R2 late-update Euclidean control; standard geometry, not a new decoding principle.')
    save('results/r18_PILOT_ANALYSIS.json',result);print(result['status'],counts,screens,flush=True)


if __name__=='__main__':main()
