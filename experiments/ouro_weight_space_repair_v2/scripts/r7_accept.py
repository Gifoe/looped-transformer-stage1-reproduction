"""Post-lock CPU acceptance: fresh R6 must equal frozen R6 per question."""
import numpy as np
from r7_core import read,sha,jsonlines,save


def main():
    from r7_evaluate import rows
    from slcd_r6_scores import selected_score
    lock=read('configs/r7_FINAL_METHOD_LOCK.json');assert lock['test_access_authorized']
    oldlock=read('configs/slcd_R6_FINAL_LOCK.json');oldpath='results/slcd_r6_priors_test.jsonl'
    receipt=read(oldpath.replace('.jsonl','_COMPLETE.json'))
    assert receipt['status']=='complete' and sha(oldpath)==receipt['score_file_sha256']
    assert receipt['provenance']['authorization_lock_sha256']==sha('configs/slcd_R6_FINAL_LOCK.json')
    assert sha('results/slcd_original_test.jsonl')==oldlock['original_test_scores_sha256']
    prior={r['id']:r for r in jsonlines(oldpath)};original={r['id']:r for r in jsonlines('results/slcd_original_test.jsonl')}
    data=rows('test');assert len(data)==1172 and {r['id'] for r in data}==set(prior)==set(original)
    error=0.
    for row in data:
        p=prior[row['id']];q=original[row['id']]
        assert p['signature']==receipt['signature']
        assert row['gold']==q['gold']==p['gold'] and row['prompt_sha256']==q['prompt_sha256']==p['prompt_sha256']
        expected=selected_score(q['methods'],p['prior_methods'],'r6_prior_projection')
        difference=float(np.max(np.abs(expected-np.asarray(row['baseline_scores']['M0_R6']))));error=max(error,difference)
        assert difference==0.,('R6 per-question baseline differs',row['id'],difference)
        for name,value in row['prior_methods'][0].items():assert value['sum_scores']==p['prior_methods'][name]['sum_scores'],('N0 test sum differs',row['id'],name)
    save('results/r7_TEST_ACCEPTANCE.json',dict(status='passed',n=1172,max_R6_candidate_score_abs_error=error,
        N0_prior_sums_exact=True,frozen_R6_prior_receipt_sha256=sha(oldpath.replace('.jsonl','_COMPLETE.json')),
        final_method_lock_sha256=sha('configs/r7_FINAL_METHOD_LOCK.json'),new_score_file_sha256=sha('results/r7_test_final.jsonl'),
        scope='Exact complete per-question baseline scores and N0 sums, not only equal aggregate accuracy'))
    print('R7_TEST_ACCEPTANCE_PASSED',flush=True)


if __name__=='__main__':main()
