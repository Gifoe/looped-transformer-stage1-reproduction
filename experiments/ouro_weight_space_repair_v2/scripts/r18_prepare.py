"""Freeze exactly one uniform-metric candidate and its own four controls."""
from r18_common import ROOT,LOCK,read,sha,save


def main():
    assert not any((ROOT/p).exists() for p in [LOCK,'results/r18_pilot.jsonl','logs/r18_gpu_budget.json'])
    old=read('configs/r16_NATIVE_KL_PILOT_LOCK.json');cohort=old['cohort'];assert len(cohort)==64
    ledgers=['logs/r7_gpu_budget.json','logs/r9_gpu_budget.json','logs/r10_gpu_budget.json','logs/r11_gpu_budget.json','logs/r15_gpu_budget.json','logs/r16_gpu_budget.json','logs/r17_gpu_budget.json']
    hashes={p:sha(p) for p in ledgers};total=sum(read(p)['used_seconds'] for p in ledgers);assert abs(total-17785.147090673447)<1e-6
    sources=[p for p in old['source_sha256'] if not p.startswith('r16_')]+[p.name for p in sorted((ROOT/'scripts').glob('r18_*.py'))]
    spec=dict(status='frozen_bounded_development_pilot',new_method_candidates=1,heldout_access_authorized=False,external_benchmarks_authorized=False,cohort=cohort,
        cohort_selection='Exactly the original seed2021 uniform64 from R16; no new outcome-based sampling, not independent validation. Same development inputs and prior known baseline outcomes.',
        source_sha256={p:sha('scripts/'+p) for p in sources},input_sha256=old['input_sha256'],model_identity=old['model_identity'],
        hypothesis='R17 high fixed-shuffle alignment motivates checking native Fisher concentration. This neither proves Fisher is wrong nor reference semantics are shared. Test a single standard uniform-metric ablation, no temperature or geometry mixture.',
        method=dict(name='uniform_projection',references=['N0'],g='.5*(q4-q1), FP32; original Fixed F=q4+g',e='.5*(N0_4-N0_1), same literal candidate prefix and independent encode_pair',
            metric='Uniform over full vocabulary, center by arithmetic mean in FP64.',alpha='clip(mean(center_U(g)*center_U(e))/mean(center_U(e)^2),0,1); denominator <=64eps64*maxabs(center_U(e))^2 ->0.',
            output='F-alpha*center_U(e) inFP64 ->FP32; full-vocab FP32 log_softmax, concatenate target LP then original FP32 sum, one original char denominator, unchanged Q4-Q5. No second R6 subtraction.',
            novelty='Standard Euclidean projection. Old R2 full-vocabulary Euclidean late-update controls existed; this is a different fixed prior-direction ablation, not first use of this geometry.'),
        controls=dict(norm_U='q4+rho_U*g, rho_U=||center_U(g-alpha*eU)||U/||center_U(g)||U, no clipping.',norm_P='Same actual uniform correction, centered in original pQ4, match residual guidance P norm, rho_P may exceed1, no clipping.',
            shuffle_U='One seed2021 full-vocab permutation of actual uniform correction; re-center and match uniform correction norm.',shuffle_P='Same fixed permutation of actual correction centered in native pQ4; re-center and match actual P correction norm.',
            eligibility='All four are controls, never promotable; norm identities are before finalFP32 rounding.'),
        diagnostic=dict(effective_rank='H=diag(pQ4)-pQ4*pQ4T; rank_eff=tr(H)^2/tr(H²). Stable positive diagonal/offdiagonal sum, special max-coordinate complement for cancellation.',
            numerical_rule='Rank undefined when tr(H)<=1e-8 or tr(H²)<=0; preserve mask and zero placeholder. Diagnostic only: never gates/scales decoding.',
            other='Probability IPR, pmax, entropy, original g/e cosine under P/U and one fixed shuffle, and actual correction energy under BOTH metrics for old/new methods.',
            restriction='Not a rerun of R17 cue-response alignments; no changed-cue activation is available here. Low rank is not proof of causal failure, and shuffle is not isotropic.'),
        numerical=dict(real='Original main/N0 native/Fixed scores and old M2 alpha/full scores exact0 parity. First real position NumPy/GPU alpha<=2^-32, FP32logit<=2e-5. Four pre-FP32 norm identities <=1e-10.',
            synthetic='Independent NumPy primary, exact zero Fixed, identical-reference collapse, explicit small covariance rank and local-vs-global direction witness.'),
        screen=dict(require_full64=True,benefit='Strictly beats R6, old M2, both norm controls and both shuffle controls; retain ALL pilot R6-only-vs-Fixed successes.',
            on_failure='Close uniform-metric route. No temperature, mixture, tail truncation, sign flip, extra reference, strength sweep or selection of controls.',
            on_pass='Only feasibility review for complete development/confirmation/validation/test and reference stability within remaining ORIGINAL budget; no automatic extension or new test.',
            inference='Paired exact McNemar descriptive only; reused64 is not independent evidence or a proof of low-rank causality.'),
        budget=dict(original_cap_seconds=21600,inherited_ledger_sha256=hashes,inherited_total_seconds=total,limit_seconds=120,termination_grace_seconds=15,
            original_complete_ARC_worker_seconds_already_used=5625.518387556076,
            scope='One bounded post-evaluation metric pilot, all actual worker/load/failure time included. Seven old ledgers unchanged, no reset. Shared GPU same mutex+CPU nice15, no other processes altered, no automatic retry.'))
    save(LOCK,spec);print('R18_FROZEN',sha(LOCK),total)


if __name__=='__main__':main()
