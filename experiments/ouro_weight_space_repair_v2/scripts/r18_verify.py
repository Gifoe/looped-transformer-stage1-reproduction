"""Read-only replay of complete-pilot classifications, scalar identities and budget."""
import csv,hashlib,json,math
import numpy as np
from r18_common import ROOT,LOCK,read,sha,digest,jsonlines,gate


def main():
    spec=gate();done=read('results/r18_pilot_COMPLETE.json');analysis=read('results/r18_PILOT_ANALYSIS.json');sup=read('logs/r18_SUPERVISOR.json')
    assert done['status']==sup['status']=='complete' and done['n']==64 and not done['weights_modified'] and not done['heldout_accessed']
    assert done['protocol_sha256']==analysis['protocol_sha256']==sha(LOCK) and done['head_norm_before']==done['head_norm_after']
    assert sha(done['score_file'])==done['score_file_sha256']
    records=list(jsonlines(done['score_file']));assert [r['id'] for r in records]==[r['id'] for r in spec['cohort']]
    assert len(records)==len({r['id'] for r in records})==64
    old={r['id']:r for split in ['discovery','selection'] for r in jsonlines('results/r7_'+split+'_development.jsonl')}
    methods=list(analysis['counts']);counts={k:0 for k in methods};paired={k:[0,0] for k in methods if k!='uniform_projection'};retained=0;special=0;positions=0;maxnorm=0.;max_reconstruction=0.
    for row,chosen in zip(records,spec['cohort']):
        assert row['signature']==done['signature']==digest(dict(protocol_sha256=sha(LOCK),cohort=spec['cohort'])) and row['cohort_entry_sha256']==digest(chosen)
        original=old[row['id']];y=int(original['gold']);assert len(row['candidates'])==len(original['method_scores']['M2_N0_token'])
        scores={k:[c['scores'][k] for c in row['candidates']] for k in methods[:6]}
        assert scores['old_Fisher']==original['method_scores']['M2_N0_token']
        for k,b in [('R6','M0_R6'),('fixed','fixed'),('A1','A1'),('native4','native4')]:scores[k]=original['baseline_scores'][b]
        pred={k:max(range(len(v)),key=lambda j:v[j]) for k,v in scores.items()};correct={k:v==y for k,v in pred.items()}
        for k in methods:counts[k]+=correct[k]
        for k in paired:paired[k][0]+=correct['uniform_projection'] and not correct[k];paired[k][1]+=correct[k] and not correct['uniform_projection']
        if correct['R6'] and not correct['fixed']:special+=1;retained+=correct['uniform_projection']
        for c in row['candidates']:
            assert c['parity_error']==0 and c['characters']>0
            m={k:np.asarray(v) for k,v in c['metrics'].items()};positions+=c['tokens'];assert all(len(v)==c['tokens'] for v in m.values())
            a=m['alpha'];assert np.all((a>=0)&(a<=1))
            den=m['uniform_denominator'];num=m['uniform_numerator'];bad=den<=m['uniform_threshold'];expected_a=np.clip(np.divide(num,den,out=np.zeros_like(num),where=~bad),0,1)
            assert np.max(np.abs(expected_a-a))<1e-12
            for one,two in [('matched_U_norm2','residual_U_norm2'),('matched_P_norm2','residual_P_norm2'),('shuffle_U_norm2','correction_U_norm2'),('shuffle_P_norm2','correction_P_norm2')]:maxnorm=max(maxnorm,float(np.max(np.abs(m[one]-m[two]))))
            assert maxnorm<=1e-10
            valid=m['Fisher_rank_defined'].astype(bool);rank=np.divide(m['Fisher_trace']**2,m['Fisher_trace_square'],out=np.zeros_like(a),where=valid)
            assert np.max(np.abs(rank-m['Fisher_effective_rank']))<1e-10 and np.array_equal(valid,(m['Fisher_trace']>1e-8)&(m['Fisher_trace_square']>0))
            for k,total in c['sums'].items():
                assert c['scores'][k]==total/c['characters']+c['late']
                lp=m[k+'_LP'];error=abs(math.fsum(lp.tolist())-total);bound=64*np.finfo(np.float32).eps*max(1.,math.fsum(abs(x) for x in lp));assert error<=bound
                max_reconstruction=max(max_reconstruction,error)
                # Account for FP32 subtraction rounding in stored decomposition.
                decomp=m[k+'_target_shift']-m[k+'_normalizer_change']-m[k+'_LP_change'];scale=1+np.abs(m[k+'_target_shift'])+np.abs(m[k+'_normalizer_change'])+np.abs(m[k+'_LP_change'])
                assert np.all(np.abs(decomp)<=8*np.finfo(np.float32).eps*scale)
    assert counts==analysis['counts'] and analysis['R6_unique_vs_Fixed']==dict(total=special,retained=retained)
    for k,(r,d) in paired.items():
        n=r+d;p=1. if n==0 else min(1.,sum(math.comb(n,i) for i in range(min(r,d)+1))*2/2**n)
        assert analysis['paired'][k]==dict(rescues=r,damages=d,net=r-d,exact_two_sided_McNemar_p=p)
    screens=[counts['uniform_projection']>counts[k] for k in ['R6','old_Fisher','matched_guidance_U','matched_guidance_P','shuffled_correction_U','shuffled_correction_P']]+[retained==special]
    assert list(analysis['screens'].values())==screens and analysis['eligible_for_feasibility_review']==all(screens) and not analysis['automatic_extension_authorized']
    ledger=read('logs/r18_gpu_budget.json');inherited=sum(read(p)['used_seconds'] for p in spec['budget']['inherited_ledger_sha256'])
    for p,h in spec['budget']['inherited_ledger_sha256'].items():assert sha(p)==h
    total=inherited+ledger['used_seconds'];assert ledger['inherited_total_seconds']==inherited and ledger['total_used_seconds']==total<=21600 and ledger['used_seconds']==sup['gpu_worker_seconds']
    return dict(status='passed_read_only_independent_replay',n=64,positions=positions,counts=counts,eligible_for_feasibility_review=all(screens),pre_FP32_control_norm_max_error=maxnorm,
        scalar_FP32_sum_reconstruction_max_error=max_reconstruction,aggregate_worker_seconds=total,remaining_worker_seconds=21600-total,heldout_accessed=False,external_benchmarks_started=False,
        scope='Classification/statistical/accounting replay; no second model run and no independent full-vocabulary activation archive. Control norm identity is before final FP32 rounding.')


if __name__=='__main__':print(json.dumps(main(),indent=2))
