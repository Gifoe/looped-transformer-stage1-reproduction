"""Freeze one label-independent pilot before any new model effects exist."""
import json
import numpy as np
from r16_common import ROOT,LOCK,read,sha,jsonlines,save


def main():
    assert not any((ROOT/p).exists() for p in [LOCK,'results/r16_pilot.jsonl','logs/r16_gpu_budget.json'])
    old=read('configs/r15_CONTEXT_PROBE_LOCK.json')
    ordered=[dict(id=r['id'],split=split) for split in ['discovery','selection'] for r in jsonlines('data/slcd_'+split+'.jsonl')]
    assert len(ordered)==len({r['id'] for r in ordered})==576
    cohort=[ordered[int(j)] for j in np.random.default_rng(2021).choice(len(ordered),size=64,replace=False)]
    inherited={p:sha(p) for p in ['logs/r7_gpu_budget.json','logs/r9_gpu_budget.json','logs/r10_gpu_budget.json','logs/r11_gpu_budget.json','logs/r15_gpu_budget.json']}
    total=sum(read(p)['used_seconds'] for p in inherited);assert abs(total-17656.16580247879)<1e-6
    sources=[p for p in old['source_sha256'] if not p.startswith('r15_')]+[p.name for p in sorted((ROOT/'scripts').glob('r16_*.py'))]
    spec=dict(status='frozen_bounded_development_pilot',new_method_candidates=1,heldout_access_authorized=False,external_benchmarks_authorized=False,
        cohort=cohort,cohort_selection='Uniform64 without replacement from original ordered discovery384 then selection192; NumPy default_rng(2021).choice. No gold, prediction, margin or R15 feature used to choose IDs. These development data were previously reused: no independent-validation claim.',
        source_sha256={p:sha('scripts/'+p) for p in sources},input_sha256=old['input_sha256'],model_identity=old['model_identity'],
        method=dict(name='exact_native_KL',references=['N0'],p0='native Q4 softmax FP64',guidance='g=.5*(q4-q1), F=q4+g literal FP32',
            e='center_p0(.5*(N0_4-N0_1)), original independently encoded continuation literal prefix',
            objective='f(a)=KL(p0||softmax(F-a*e))-KL(p0||softmax(F)); general helper also supports explicitly specified .5*a^2*U, but pilot is J1 U=0.',
            constraint='a in [0,1]; fprime(0)>=0 ->0, fprime(1)<=0 ->1, otherwise32 fixed FP64 bisections. Flat e uses64eps64 predeclared rule ->0.',
            output='F-a*e ->FP32, full-vocabulary FP32 log_softmax, concatenate all target LP then original FP32 sum, divide once by original char_length, add unchanged Q4-Q5. Never subtract R6 again.',
            interpretation='Standard convex native-distribution KL calibration, not a new mathematical principle and no correctness guarantee. Native Q4 may itself be wrong. Endpoint solution need not match moments.'),
        controls=dict(norm='q4+rho*g where rho=||center(g-a*e)||_p0/||center(g)||_p0, NOT clipped; final FP32. a0 uses exact Fixed.',
            shuffle='One seed2021 fixed full-vocabulary permutation, applied to actual correction a*e; re-center and match actual correction norm. Never an eligible method.',
            baselines=['R6','old_M2_Fisher','Fixed','A1','native4'],zero_repair='Exact literal Fixed logits and A1 final score, not R6.'),
        numerical=dict(synthetic='Independent NumPy/Torch CPU and synthetic grid check, no dataset coefficient grid.',
            real='All original main/ref native and Fixed scores exact0 error; old M2 coefficients and full scores exact0 error; first real token NumPy vsGPU alpha<=2^-32, FP32logit<=2e-5.',
            objective='Pre-round optimum <=0 and <= old Fisher objective within128*eps64*V*(1+maxabs(F-F_first)+maxabs(e)). FP32 objective <=ideal+2*maxlogitrounding+same bound.',
            control_norm_absolute_error_limit=1e-10),
        screen=dict(require_full64=True,benefit='Strictly more correct than R6, old M2, own norm control and own shuffle control; retain ALL pilot R6-only-vs-Fixed successes.',
            inference='Report exact paired McNemar, no significance/novelty claim from pilot.',
            on_failure='Stop this objective. No scale/sign/reference-set change or promotion of controls; no heldout or external tasks.',
            on_pass='Feasibility review only, not automatic extension. Full development/confirmation/validation/test budgets and reference stability remain unresolved. Original reserve not silently relaxed.'),
        budget=dict(original_cap_seconds=21600,inherited_ledger_sha256=inherited,inherited_total_seconds=total,limit_seconds=120,termination_grace_seconds=15,
            original_complete_ARC_worker_seconds_already_used=5625.518387556076,
            scope='One post-evaluation approximation audit; actual worker/load/failure time charged under existing total cap. No budget reset, shared GPU low CPU priority, no other processes altered. No automatic retry. Partial results never used for method screen.'),
        timing_basis=dict(cached_main=True,N0_all_choices_historical_seconds=.29564949745933217,solver_cost='Unmeasured; this pilot measures it, not standalone deployment throughput.'),
        scope='One falsifiable candidate after R15 association failure, not a new frozen primary. No heldout label/row or external dataset access.')
    save(LOCK,spec);print('FROZEN',sha(LOCK),len(cohort),total)


if __name__=='__main__':main()
