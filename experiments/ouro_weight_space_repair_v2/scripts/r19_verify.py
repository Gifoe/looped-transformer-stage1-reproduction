"""Read-only replay of complete-pilot classifications, scalar identities and budget."""
import csv,hashlib,json,math
import numpy as np
from r19_common import ROOT,LOCK,read,sha,digest,jsonlines,gate


def main():
    spec=gate();done=read('results/r19_pilot_COMPLETE.json');analysis=read('results/r19_PILOT_ANALYSIS.json');sup=read('logs/r19_SUPERVISOR.json');model_config=read('artifacts/r19_MODEL_CONFIG.json');vocab_size=int(model_config['vocab_size']);assert sha('artifacts/r19_MODEL_CONFIG.json')==spec['model_identity']['metadata_and_tokenizer_sha256']['config.json']
    assert done['status']==sup['status']=='complete' and done['n']==64 and not done['weights_modified'] and not done['heldout_accessed']
    assert done['protocol_sha256']==analysis['protocol_sha256']==sha(LOCK) and done['head_norm_before']==done['head_norm_after']
    assert sha(done['score_file'])==done['score_file_sha256']
    records=list(jsonlines(done['score_file']));assert [r['id'] for r in records]==[r['id'] for r in spec['cohort']]
    assert len(records)==len({r['id'] for r in records})==64
    old={r['id']:r for split in ['discovery','selection'] for r in jsonlines('results/r7_'+split+'_development.jsonl')}
    native3={r['id']:r for split in ['discovery','selection'] for r in jsonlines('results/slcd_r2_'+split+'_development.jsonl')}
    depth_checks=[]
    methods=list(analysis['counts']);counts={k:0 for k in methods};paired={k:[0,0] for k in methods if k!='paired_loop_protection'};retained=0;special=0;positions=0;maxnorm=0.;max_reconstruction=0.
    for row,chosen in zip(records,spec['cohort']):
        assert row['signature']==done['signature']==digest(dict(protocol_sha256=sha(LOCK),cohort=spec['cohort'])) and row['cohort_entry_sha256']==digest(chosen)
        original=old[row['id']];y=int(original['gold']);assert len(row['candidates'])==len(original['method_scores']['M2_N0_token'])
        scores={k:[c['scores'][k] for c in row['candidates']] for k in methods[:6]}
        assert scores['old_Fisher']==original['method_scores']['M2_N0_token']
        for k,b in [('R6','M0_R6'),('fixed','fixed'),('A1','A1'),('native4','native4')]:scores[k]=original['baseline_scores'][b]
        pred={k:max(range(len(v)),key=lambda j:v[j]) for k,v in scores.items()};correct={k:v==y for k,v in pred.items()}
        for k in methods:counts[k]+=correct[k]
        for k in paired:paired[k][0]+=correct['paired_loop_protection'] and not correct[k];paired[k][1]+=correct[k] and not correct['paired_loop_protection']
        if correct['R6'] and not correct['fixed']:special+=1;retained+=correct['paired_loop_protection']
        for c in row['candidates']:
            assert c['parity_error']==0 and c['characters']>0
            m={k:np.asarray(v) for k,v in c['metrics'].items()};positions+=c['tokens'];assert all(len(v)==c['tokens'] for v in m.values())
            assert c['original_native3_sum']==native3[row['id']]['original_methods']['native_k3']['sum_scores'][c['choice']]
            if c['independent_depth3_check'] is not None:depth_checks.append(c['independent_depth3_check'])
            a=m['alpha'];assert np.all((a>=0)&(a<=1))
            defined=m['orthogonality_defined'].astype(bool)
            assert np.all(np.abs(m['orthogonality_pre'][defined])<=m['orthogonality_arithmetic_bound'][defined]) and np.all(np.abs(m['orthogonality_rounded'])<=m['orthogonality_bound'])
            bound=128*np.finfo(float).eps*vocab_size*(1+m['e_norm2']+m['protected_norm2'])
            assert np.all(np.abs(m['protected_norm2']-m['protection_b']**2*m['s_norm2'])<=bound)
            assert np.all(np.abs(m['residual_prior_norm2']-(m['e_norm2']-m['protected_norm2']))<=bound)
            maxnorm=max(maxnorm,float(np.max(np.abs(m['matched_guidance_norm2']-m['residual_norm2']))),float(np.max(np.abs(m['shuffled_correction_norm2']-m['correction_norm2']))));assert maxnorm<=1e-10
            for k,total in c['sums'].items():
                assert c['scores'][k]==total/c['characters']+c['late']
                lp=m[k+'_LP'];error=abs(math.fsum(lp.tolist())-total);bound=64*np.finfo(np.float32).eps*max(1.,math.fsum(abs(x) for x in lp));assert error<=bound
                max_reconstruction=max(max_reconstruction,error)
                # Account for FP32 subtraction rounding in stored decomposition.
                decomp=m[k+'_target_shift']-m[k+'_normalizer_change']-m[k+'_LP_change'];scale=1+np.abs(m[k+'_target_shift'])+np.abs(m[k+'_normalizer_change'])+np.abs(m[k+'_LP_change'])
                assert np.all(np.abs(decomp)<=8*np.finfo(np.float32).eps*scale)
    assert len(depth_checks)==1 and all(depth_checks[0].values()) and depth_checks[0]==analysis['independent_depth3_checks']
    assert counts==analysis['counts'] and analysis['R6_unique_vs_Fixed']==dict(total=special,retained=retained)
    for k,(r,d) in paired.items():
        n=r+d;p=1. if n==0 else min(1.,sum(math.comb(n,i) for i in range(min(r,d)+1))*2/2**n)
        assert analysis['paired'][k]==dict(rescues=r,damages=d,net=r-d,exact_two_sided_McNemar_p=p)
    screens=[counts['paired_loop_protection']>counts[k] for k in ['R6','old_Fisher','reference_only_axis','matched_guidance','shuffled_correction','shuffled_axis']]+[retained==special]
    assert list(analysis['screens'].values())==screens and analysis['eligible_for_feasibility_review']==all(screens) and not analysis['automatic_extension_authorized']
    ledger=read('logs/r19_gpu_budget.json');inherited=sum(read(p)['used_seconds'] for p in spec['budget']['inherited_ledger_sha256'])
    for p,h in spec['budget']['inherited_ledger_sha256'].items():assert sha(p)==h
    total=inherited+ledger['used_seconds'];assert ledger['inherited_total_seconds']==inherited and ledger['total_used_seconds']==total<=21600 and ledger['used_seconds']==sup['gpu_worker_seconds']
    return dict(status='passed_read_only_independent_replay',n=64,positions=positions,counts=counts,eligible_for_feasibility_review=all(screens),pre_FP32_control_norm_max_error=maxnorm,
        scalar_FP32_sum_reconstruction_max_error=max_reconstruction,aggregate_worker_seconds=total,remaining_worker_seconds=21600-total,heldout_accessed=False,external_benchmarks_started=False,
        scope='Classification/statistical/accounting replay; no second model run and no independent full-vocabulary activation archive. Control norm identity is before final FP32 rounding.')


if __name__=='__main__':print(json.dumps(main(),indent=2))
