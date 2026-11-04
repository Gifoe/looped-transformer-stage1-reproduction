"""Read-only verification of released development-only evidence, no model load."""
import csv,json,math
import numpy as np
from r7_core import ROOT,read,sha,jsonlines


def norm(x):
    v=np.asarray(x,dtype=np.float64);v=v-v[0];v=v-math.fsum(v)/len(v)
    return math.sqrt(math.fsum(float(z)*float(z) for z in v))


def main():
    p13=read('configs/r13_PREFIX_AUDIT_PROTOCOL.json');p14=read('configs/r14_ANCHOR_DIAGNOSTIC_PROTOCOL.json')
    a13=read('results/r13_PREFIX_AUDIT_COMPLETE.json');a14=read('results/r14_development_COMPLETE.json')
    assert sha('scripts/r13_prefix_audit.py')==p13['source_sha256'] and sha('scripts/r14_anchor_diagnostic.py')==p14['source_sha256']
    assert sha('configs/r13_PREFIX_AUDIT_PROTOCOL.json')==a13['protocol_sha256'] and sha('configs/r14_ANCHOR_DIAGNOSTIC_PROTOCOL.json')==a14['protocol_sha256']
    assert a13==read('artifacts/r13_SERVER_CPU_AUDIT_COMPLETE.json'),'CPU platforms disagree beyond file line endings'
    assert all(sha(p)==h for p,h in p13['input_sha256'].items())
    assert all(sha('scripts/'+p)==h for p,h in p14['inference_source_sha256'].items())
    for p,h in a13['outputs'].items():assert sha(p)==h
    authority={}
    for split in ['discovery','selection']:
        path=f'results/r7_{split}_development.jsonl';assert sha(path)==p14['input_sha256'][path]
        for row in jsonlines(path):authority[row['id']]=row
    data=list(jsonlines(a14['score_file']));assert sha(a14['score_file'])==a14['score_file_sha256'] and len(authority)==len(data)==len({r['id'] for r in data})==576
    max_norm_error=0.;lp_sums_nonpositive=True;positions=0
    for row in data:
        old=authority[row['id']];assert row['gold']==old['gold'] and row['protocol_sha256']==a14['protocol_sha256']
        r=np.asarray(old['baseline_scores']['M0_R6']);m=np.asarray(old['method_scores']['M2_N0_token']);b=np.asarray(old['method_scores']['C_R6alpha_token']);v=row['scores'];a=np.asarray(v['A1'])
        assert np.array_equal(v['anchored_token_refinement'],r+(m-b)) and np.array_equal(v['reversed_refinement'],r-(m-b))
        assert np.array_equal(v['R6'],r) and np.array_equal(v['M2'],m) and np.array_equal(v['bridge'],b)
        size=norm(np.asarray(v['anchored_token_refinement'])-a);control=norm(np.asarray(v['prior_norm'])-a)
        max_norm_error=max(max_norm_error,abs(size-control));assert abs(size-control)<1e-12*max(size,1.)
        assert abs(norm(np.asarray(v['shuffled_refinement'])-r)-norm(m-b))<1e-12*max(norm(m-b),1.)
        for name,vector in v.items():assert row['predictions'][name]==int(np.argmax(vector)) and row['correct'][name]==(int(np.argmax(vector))==row['gold'])
        q4=np.asarray(old['original_methods']['native_k4']['normalized_scores']);q5=np.asarray(old['original_methods']['native_k5']['normalized_scores']);late=q4-q5
        for method in p13['methods']:
            total=(np.asarray(old['method_scores'][method])-late)*old['choice_char_lengths']
            lp_sums_nonpositive &= bool(np.all(total<=0))
        lp_sums_nonpositive &= all(z<=0 for z in old['original_methods']['loopcd_k4_fixed']['sum_scores'])
        positions+=sum(old['token_counts'])*len(p13['methods'])
    assert lp_sums_nonpositive and positions==a13['all_prediction_positions']==40251
    for result in a14['comparisons']:
        name=result['reference'];x=[r['correct']['anchored_token_refinement'] for r in data];b=[r['correct'][name] for r in data]
        rescue=sum(a and not c for a,c in zip(x,b));damage=sum(not a and c for a,c in zip(x,b));n=rescue+damage
        p=1. if n==0 else min(1.,2*sum(math.comb(n,k) for k in range(min(rescue,damage)+1))/2**n)
        assert result['correct']==sum(x)==349 and result['reference_correct']==sum(b) and result['net']==rescue-damage
        assert (result['rescue'],result['damage'],result['exact_mcnemar_p'])==(rescue,damage,p)
    unique=[r for r in data if r['correct']['R6'] and not r['correct']['fixed']]
    assert len(unique)==a14['total_R6_special']==21 and sum(r['correct']['anchored_token_refinement'] for r in unique)==a14['retained_R6_special']==12
    assert not a14['passes_gate'] and a14['criteria']['minimum_net'] is False and not a14['criteria']['paired_R6_p']
    assert not any((ROOT/p).exists() for p in ['results/r14_confirmation_COMPLETE.json','results/r14_confirmation_scores.jsonl'])
    assert not a13['heldout_accessed'] and not a14['heldout_accessed'] and a13['GPU_seconds']==a14['GPU_seconds']==0
    used=sum(read('logs/'+name+'_gpu_budget.json')['used_seconds'] for name in ['r7','r9','r10','r11'])
    assert abs(used-17479.398183107376)<1e-6 and used<21600
    print(json.dumps(dict(status='verified_complete_development_only_evidence',n=576,positions=positions,anchor_correct=349,R6_correct=360,
        max_independent_norm_error=max_norm_error,R13_cross_platform_CSV_bytes_exact=True,heldout_accessed=False,new_GPU_seconds=0,
        original_GPU_worker_used_seconds=used,remaining_GPU_worker_seconds=21600-used,goal_success_declared=False)),flush=True)


if __name__=='__main__':main()
