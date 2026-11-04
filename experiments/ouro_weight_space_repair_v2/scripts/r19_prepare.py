"""Freeze one same-prompt last-loop protection and its reference-only falsifier."""
from r19_common import ROOT,LOCK,read,sha,save


def main():
    assert not any((ROOT/p).exists() for p in [LOCK,'results/r19_pilot.jsonl','logs/r19_gpu_budget.json'])
    old=read('configs/r18_UNIFORM_METRIC_PILOT_LOCK.json');ledgers=list(old['budget']['inherited_ledger_sha256'])+['logs/r18_gpu_budget.json'];hashes={p:sha(p) for p in ledgers};total=sum(read(p)['used_seconds'] for p in ledgers)
    assert abs(total-17870.32302212715)<1e-6 and len(ledgers)==8
    sources=[p for p in old['source_sha256'] if not p.startswith('r18_')]+[p.name for p in sorted((ROOT/'scripts').glob('r19_*.py'))]
    inputs=old['input_sha256'].copy()
    for s in ['discovery','selection']:
        for suffix in ['.jsonl','_COMPLETE.json']:p='results/slcd_r2_'+s+'_development'+suffix;inputs[p]=sha(p)
    spec=dict(status='frozen_bounded_development_pilot',new_method_candidates=1,heldout_access_authorized=False,external_benchmarks_authorized=False,cohort=old['cohort'],
        cohort_selection='Same original seed2021 uniform64 as R16/R18; repeatedly used development, no outcome-based resampling and no independent-validation claim.',
        source_sha256={p:sha('scripts/'+p) for p in sources},input_sha256=inputs,model_identity=old['model_identity'],
        hypothesis='Does preserving the real-question-minus-N0 final trained-loop innovation help more than preserving reference-only innovation? This is not an assumed semantic axis and axis sign is uninformative about correctness.',
        method=dict(name='paired_loop_protection',references=['N0'],metric='Original fixed native pQ4 full-vocabulary FP64 Fisher inner product, NOT uniform.',
            g='.5*(Q4-Q1) FP32, original Fixed F=Q4+g',e='center_p(.5*(N0_4-N0_1))',axis='s=center_p((Q4-Q3).double-(N0_4-N0_3).double), with individual loop differences FP32.',
            protect='b=<e,s>/||s||², eperp=e-b*s; if axis norm<=64eps64*maxabs(s)^2, exact originalM2.',
            alpha='clip(<center_p(g),eperp>/||eperp||²,0,1); residual norm<=64eps64*maxabs(e)^2 ->0.',
            output='F-alpha*eperp FP64 ->FP32; actual full-vocabulary FP32 log_softmax, full continuation FP32 sum, one original char denominator, unchanged Q4-Q5; no second R6 subtraction.',
            scope='New loop3 direction information, standard projection family. Not a new principle, reliable-semantic direction or already reference-stable repair.'),
        controls=dict(reference_only='Use v=center_p(N0_4-N0_3) instead of s; otherwise identical. Mandatory ineligible falsifier of reference-self coupling.',
            matched_guidance='Match current residual guidance P norm by q4+rho*g with no clipping. Pre-FP32 identity.',
            shuffled_correction='Permute actual completed correction using one seed2021 vocabulary permutation; re-center and match correction P norm.',
            shuffled_axis='Permute s with the same fixed permutation, re-center; recompute protection and alpha. Not norm-matched, energy reported. Distinct from completed-correction shuffle.',
            eligibility='All four are controls and cannot be selected. No alternate loop indices, axis combinations or sign tuning after feedback.'),
        numerical=dict(native3='Every cached native3 sum exactly matches original R2 native3 scores, same model/prompt. First candidate independently forward Q todepth3 and N0 todepth3, require full-vocabulary head tensors exactly equal cached6/shared4 third-loop readouts.',
            existing='All original main/N0 native/Fixed and old M2 alpha/full scores exact0 parity. First real NumPy/GPU alpha<=2^-32 and FP32logit<=2e-5.',
            projection='Resolved axes: absolute pre-rounding protection dot<=256eps64*V*(1+maxabs(correction)*maxabs(s)). Rounded dot bounded by pre-dot+maxround*E_p|s|+same arithmetic bound. Unresolved axes make no orthogonality claim.',
            controls='Two pre-FP32 norm identities<=1e-10. Independent synthetic NumPy primary, zeroFixed, no-axis exactM2, axis sign invariance, gauge, identical reference/batch extension.',
            diagnostic='Axis fraction outside numerically resolved span(g,e) is descriptive only; never gates or scales decoding.'),
        screen=dict(require_full64=True,benefit='Strictly exceeds R6,oldM2,reference-only axis,norm control,completed-correction shuffle,and recomputed-axis shuffle; retain ALL pilot R6-only-vs-Fixed successes.',
            on_failure='Stop this protected-last-loop route. User requests next design by Astra based on results, then execution by GPT-6.1 Sol. Do not retune loop/range/axis to save this candidate.',
            on_pass='Feasibility review only; no automatic complete test or J3 run. Full development/confirmation/validation/ARC and reference stability must fit original remaining budget or require concrete budget review.',
            inference='Exact paired McNemar descriptive; reused64 not independent confirmation.'),
        budget=dict(original_cap_seconds=21600,inherited_ledger_sha256=hashes,inherited_total_seconds=total,limit_seconds=120,termination_grace_seconds=15,
            original_complete_ARC_worker_seconds_already_used=5625.518387556076,
            scope='One bounded post-evaluation last-loop pilot. Actual load/checks/failure/worker time charged; eight old ledgers preserved, no reset. Same mutex, CPU nice15, no other processes altered, no automatic retry.'))
    save(LOCK,spec);print('R19_FROZEN',sha(LOCK),total)


if __name__=='__main__':main()
