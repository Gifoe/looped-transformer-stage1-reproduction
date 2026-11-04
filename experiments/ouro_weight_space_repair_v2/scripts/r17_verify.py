"""Independent read-only scalar/statistical/accounting replay; no new forwards."""
import itertools,json,math
import numpy as np
from r17_common import ROOT,LOCK,read,sha,digest,jsonlines,gate


def main():
    spec=gate();done=read('results/r17_paired_COMPLETE.json');analysis=read('results/r17_RESPONSE_ANALYSIS.json');sup=read('logs/r17_SUPERVISOR.json')
    assert done['status']==sup['status']=='complete' and done['n']==analysis['n']==12
    assert done['protocol_sha256']==analysis['protocol_sha256']==sha(LOCK) and sha(done['score_file'])==done['score_file_sha256'] and done['head_norm_before']==done['head_norm_after']
    assert not done['weights_modified'] and not done['heldout_accessed'] and not done['new_decoding_accuracy'] and not analysis['new_accuracy_results'] and not analysis['external_benchmarks_started']
    records=list(jsonlines(done['score_file']));assert [r['id'] for r in records]==[r['id'] for r in spec['cohort']]
    names=['reference','transfer','reference_shuffle','transfer_shuffle','reference_native_metric','static'];summaries=[];positions=0;maxerror=0.;complete=True
    for r,c in zip(records,spec['cohort']):
        assert r['signature']==done['signature']==digest(dict(protocol_sha256=sha(LOCK),cohort=spec['cohort'])) and r['cohort_entry_sha256']==digest(c)
        assert r['main_reference_parity_error']==0 and r['candidate']==c['candidate'] and 'gold' not in r
        m=r['metrics'];assert all(len(v)==r['tokens'] for v in m.values());positions+=r['tokens'];q={}
        for name in names:
            valid=m[name+'_defined'];values=[];squares=[]
            for i,ok in enumerate(valid):
                x,y,ip=[m[name+'_'+k][i] for k in ['a_norm2','b_norm2','inner']];assert x>=0 and y>=0
                if ok:
                    cs=ip/math.sqrt(x*y);r2=cs*cs;maxerror=max(maxerror,abs(cs-m[name+'_cosine'][i]),abs(r2-m[name+'_R2'][i]));assert abs(cs)<=1+1e-12
                    values.append(cs);squares.append(r2)
                else:assert m[name+'_cosine'][i]==m[name+'_R2'][i]==0.
            if name in names[:4]:complete&=all(valid)
            q[name+'_cosine_mean']=math.fsum(values)/len(values) if values else None;q[name+'_R2_mean']=math.fsum(squares)/len(squares) if squares else None
        for i,ok in enumerate(m['identity_error_defined']):
            x,y,ip=[m['transfer_'+k][i] for k in ['a_norm2','b_norm2','inner']]
            if ok:maxerror=max(maxerror,abs((x+y-2*ip)/(x+y)-m['identity_error'][i]))
            else:assert x+y==0 and m['identity_error'][i]==0.
        summaries.append(q)
    assert maxerror<1e-12 and positions==analysis['positions'] and complete==analysis['all_primary_and_shuffle_positions_defined']
    if complete:
        for name in names:
            for key in ['cosine','R2']:
                full=name+'_'+key+'_mean';vals=[s[full] for s in summaries]
                if any(v is None for v in vals):continue
                x=np.asarray(vals);rng=np.random.default_rng(2021);samples=x[rng.integers(0,12,size=(10000,12))].mean(-1);expected=analysis['question_summaries'][full]
                assert abs(x.mean()-expected['mean'])<1e-12 and np.max(np.abs(np.quantile(samples,[.025,.975])-expected['question_bootstrap95_CI']))<1e-12
        # Recompute sign-flip counts using frozen question means (summation/ties
        # exactly match the publicly defined NumPy analysis, not rounded CSV).
        raw=[]
        for r in records:
            raw.append({name:np.asarray(r['metrics'][name+'_R2'])[np.asarray(r['metrics'][name+'_defined'],dtype=bool)].mean() for name in names[:4]})
        for name in ['reference','transfer']:
            delta=np.asarray([r[name]-r[name+'_shuffle'] for r in raw]);observed=delta.sum();hit=0
            for signs in itertools.product([-1,1],repeat=12):hit+=np.sum(delta*np.asarray(signs))>=observed
            assert analysis['paired_shuffle_comparisons'][name]['exact_one_sided_sign_flip_p']==hit/4096
        s=analysis['question_summaries'];p=analysis['paired_shuffle_comparisons'];screen=analysis['operational_screen']
        relevant=s['reference_R2_mean']['question_bootstrap95_CI'][0]>.5 and p['reference']['exact_one_sided_sign_flip_p']<.05
        low=s['transfer_R2_mean']['question_bootstrap95_CI'][1]<.5;high=s['transfer_R2_mean']['question_bootstrap95_CI'][0]>.5 and s['transfer_cosine_mean']['question_bootstrap95_CI'][0]>0
        assert screen['reference_mean_prefix_R2_supported']==relevant and screen['transfer_mean_prefix_R2_low']==low and screen['transfer_mean_prefix_R2_positive_supported']==high
    old=sum(read(p)['used_seconds'] for p in spec['budget']['inherited_ledger_sha256'])
    for p,h in spec['budget']['inherited_ledger_sha256'].items():assert sha(p)==h
    ledger=read('logs/r17_gpu_budget.json');assert old==ledger['inherited_total_seconds'] and old+ledger['used_seconds']==ledger['total_used_seconds']<=21600 and ledger['used_seconds']==sup['gpu_worker_seconds']
    return dict(status='passed_independent_read_only_response_replay',n=12,positions=positions,scalar_identity_max_error=maxerror,question_bootstrap_and_exact_sign_flips_replayed=complete,
        aggregate_worker_seconds=ledger['total_used_seconds'],remaining_worker_seconds=21600-ledger['total_used_seconds'],operational_decision=analysis['operational_screen']['decision'],
        heldout_accessed=False,new_accuracy_result=False,new_method_candidates=0,scope='Scalar identities, question summaries and accounting; not independent new activation capture or a universal causal transport verification.')


if __name__=='__main__':print(json.dumps(main(),indent=2))
