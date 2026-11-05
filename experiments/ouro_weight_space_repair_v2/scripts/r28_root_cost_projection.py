"""CPU-only postrun cost planning. Uses measured diagnostic pipeline components
and input-only development counts; never profiles or opens heldout inputs.
"""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads((ROOT/p).read_text(encoding='utf-8'))


def main():
    c = read('results/r28_COMPLETE.json')
    book = read('logs/r28_gpu_budget.json')
    analysis = read('results/r28_ANALYSIS.json')
    dev = read('results/r28_ROOT_DEV576_RESOURCE_INVENTORY.json')
    graph = read('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json')
    assert c['status'] == 'complete' and c['forwards'] == 2371
    assert dev['questions'] == 576 and dev['boundary_admission'] and not dev['execution_boundary_counterexamples']
    pilot = graph['counts']['by_context']
    full = dev['by_context']
    ratios = dict(base=dev['unique_base_forward_count']/2366,
                  tokens=dev['depth_weighted_input_tokens']/graph['counts']['total']['depth_weighted_input_tokens'],
                  Q=full['Q']['unique_inputs']/pilot['Q']['unique_forwards'],
                  groups=full['N1']['unique_inputs']/pilot['N1_native']['unique_forwards'],
                  positions=dev['aligned_Q_position_uses']/548,
                  questions=576/24,
                  native_states=(5*full['Q']['unique_inputs']+4*sum(full[n]['unique_inputs'] for n in ('N0', 'N1', 'N2')))/6773)
    projected = {}
    factors = {}
    for category, seconds in c['component_seconds'].items():
        assert math.isfinite(seconds) and seconds >= 0
        if category.startswith(('forward_body/', 'head/')):
            operation, context, mode, purpose = category.split('/')
            if purpose != 'base':
                factor = 1.
            else:
                role = context if context in ('Q', 'N0') else context+('_CF' if mode == 'patch' else '_native')
                factor = full[context]['unique_inputs']/pilot[role]['unique_forwards']
                if operation == 'forward_body':
                    full_mean = full[context]['input_tokens']/full[context]['unique_inputs']
                    pilot_mean = pilot[role]['input_tokens']/pilot[role]['unique_forwards']
                    factor *= full_mean/pilot_mean
        elif category == 'model_load' or category.startswith('hash_and_identity/'):
            factor = 1.
        elif category.startswith(('IO/representative/', 'checks/no_hook_', 'checks/identity', 'checks/duplicate', 'checks/independent_math')):
            factor = 1.
        elif category.startswith('IO/native_states/'):
            factor = ratios['native_states']
        elif category in ('projection_and_scoring/fullV_estimators_controls', 'projection_and_scoring/normalization_buffers_diagnostics'):
            factor = ratios['Q']
        elif category in ('projection_and_scoring/reference_setup', 'checks/full_first3', 'IO/progress_json'):
            factor = ratios['groups']
        elif category in ('IO/scores_json', 'projection_and_scoring/final_FP32_sums_R6'):
            factor = ratios['positions']
        elif category == 'CPU/pre_model':
            factor = ratios['questions']
        elif category == 'checks/finite':
            factor = ratios['tokens']
        else:
            factor = ratios['base']
        factors[category] = factor
        projected[category] = seconds*factor
    residual = c['unclassified_worker_residual_before_final_receipt']*ratios['base']
    import_exit_supervisor = max(0., book['used_seconds']-c['elapsed_seconds'])
    dev_seconds = sum(projected.values())+residual+import_exit_supervisor
    # Later stages are explicitly extrapolated, not measured or parsed here.
    stages = {name: dict(questions=n, diagnostic_worker_seconds=dev_seconds*n/576,
                         status='input_counted_development' if name == 'development' else 'question_count_extrapolation_only')
              for name, n in [('development', 576), ('confirmation', 517), ('validation', 299), ('ARC_test', 1172)]}
    chain = sum(v['diagnostic_worker_seconds'] for v in stages.values())
    remaining = 21600-book['total_used_seconds']
    out = dict(status='CPU_cost_projection_not_GPU_authorization',
               pilot_actual_worker_seconds=book['used_seconds'],
               original_cap_seconds=21600, already_used_seconds=book['total_used_seconds'], remaining_seconds=remaining,
               cost_review_eligible=bool(analysis.get('cost_review_eligible', False)),
               efficacy_pass=bool(analysis.get('strong_screen', False)),
               dev_input_tasks=dev['unique_base_forward_count'], dev_input_positions=dev['aligned_Q_position_uses'],
               development_component_projection=projected, component_multipliers=factors,
               development_projected_residual=residual, fixed_import_exit_supervisor=import_exit_supervisor,
               stages=stages, chain_seconds=chain, chain_hours=chain/3600,
               planning_25percent_margin_hours=1.25*chain/3600,
               planning_50percent_margin_hours=1.50*chain/3600,
               additional_hours_at_25percent_margin=max(0., 1.25*chain-remaining)/3600,
               native_state_raw_bytes_development_upper=(5*full['Q']['unique_inputs']+4*sum(full[n]['unique_inputs'] for n in ('N0', 'N1', 'N2')))*2048*2,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               input_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                             ['results/r28_COMPLETE.json', 'logs/r28_gpu_budget.json', 'results/r28_ANALYSIS.json',
                              'results/r28_ROOT_DEV576_RESOURCE_INVENTORY.json', 'configs/r28_TRUE_PREFIX_TASK_INVENTORY.json']},
               limitations=['Diagnostic pipeline timings include required method hooks and integrity work; no uninstrumented deployment speed was measured.',
                            'Body costs are scaled by input tokens, other costs by declared task/state/group counts; hardware contention and nonlinear kernel effects are not bounded.',
                            'No confirmation, validation or test input or outcome was opened; their question-count extrapolation is not a workload guarantee.',
                            '25 and 50 percent margins are engineering scenarios, not confidence bounds; existing cached scores/states are conservatively not credited.',
                            'A cost-review flag is not efficacy evidence. No extra GPU, stage, smaller dataset or fallback is authorized by this file.'])
    with (ROOT/'results/r28_ROOT_FULL_CHAIN_COST_PROJECTION.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps({k: out[k] for k in ['cost_review_eligible', 'efficacy_pass', 'pilot_actual_worker_seconds', 'remaining_seconds', 'chain_hours', 'planning_25percent_margin_hours', 'planning_50percent_margin_hours']}))


if __name__ == '__main__':
    main()
