"""CPU-only descriptive analysis of COMPLETE discovery-layer screens.

Only reads the named new-experiment screen directory and its frozen metadata.
No validation/test inputs, model loading, fitted directions, or GPU operations.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from runtime import ROOT, save, configure
import os
configure();os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')


PATTERN = re.compile(r'^damp_(loop5|all_loops)_(attention|mlp)_layer(\d{2})$')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['status'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='layer_screen_v1')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()
    assert args.study.replace('_', '').isalnum()
    directory = ROOT/'results'/args.study
    complete = json.loads((directory/'COMPLETE.json').read_text())
    frozen = json.loads((directory/'FROZEN.json').read_text())
    assert complete['status'] == 'complete'
    scores = [json.loads(line) for line in (directory/'scores.jsonl').read_text().splitlines()]
    assert len(scores) == complete['completed']
    assert complete['implementation_sha256'] == frozen['implementation_sha256']
    ids = frozen['ids']
    by_key = {}
    for row in scores:
        assert row['scope'] == 'official_train_discovery_only' and row['cached'] is False
        assert '/train/' in row['id'] and row['id'] in ids
        assert row['implementation_sha256'] == frozen['implementation_sha256']
        key = row['id'], row['condition']
        assert key not in by_key
        by_key[key] = row
    conditions = sorted({row['condition'] for row in scores})
    assert len(conditions) == complete['conditions']
    assert all((uid, condition) in by_key for uid in ids for condition in conditions)
    native = {uid: by_key[(uid, 'noop_record_all')] for uid in ids}
    representative_path=ROOT/'configs/representative_cases.json'
    assert hashlib.sha256(representative_path.read_bytes()).hexdigest()==frozen['representative_sha256']
    representative=json.loads(representative_path.read_text())
    source_path=ROOT/'results'/representative['source_study']/'scores.jsonl'
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==frozen['native_scores_sha256']
    source={r['id']:r for r in map(json.loads,source_path.read_text().splitlines())}
    boundary_identity=[]
    for condition in conditions:
        if condition.startswith('boundary_loop5_step_'):
            coefficient=condition.removeprefix('boundary_loop5_step_')
            for uid in ids:
                actual=by_key[(uid,condition)]['methods']['native_k5']['sum_scores']
                expected=source[uid]['methods']['normalized_chord_'+coefficient]['sum_scores']
                boundary_identity.append(dict(id=uid,coefficient=coefficient,max_candidate_sum_error=max(abs(a-b) for a,b in zip(actual,expected))))
    save(str((directory/'BOUNDARY_READOUT_IDENTITY.json').relative_to(ROOT)),dict(cases=boundary_identity,passed=all(r['max_candidate_sum_error']==0 for r in boundary_identity)))
    assert all(r['max_candidate_sum_error']==0 for r in boundary_identity),'Boundary/readout identity failed; inspect before causal interpretation'
    assert not (directory/'NOOP_PARITY_FAILURE.json').exists(), 'Recorded native parity failure'
    assert all(row['native_k4_max_abs_error'] == 0 for row in native.values())
    detail = []
    for row in scores:
        match = PATTERN.match(row['condition'])
        for k in (4, 5, 6):
            value = row['methods'][f'native_k{k}']
            baseline = native[row['id']]['methods'][f'native_k{k}']
            # runner delta_margin was independently computed vs original discovery scores.
            delta = value['margin']-baseline['margin']
            assert abs(delta-row['delta_margin'][f'k{k}']) <= 1e-12
            if row['condition'] == 'noop_record_all':
                assert row['delta_margin'][f'k{k}'] == 0
            detail.append(dict(id=row['id'], group=row['group'], condition=row['condition'], k=k,
                scope='official_train_selected_representatives',
                invocation_scope=match[1] if match else 'boundary_loop5' if row['condition'].startswith('boundary') else 'noop',
                branch=match[2] if match else '', layer=int(match[3]) if match else '',
                gold=value['gold'], native_prediction=baseline['prediction'], prediction=value['prediction'],
                native_correct=baseline['correct'], correct=value['correct'],
                correction=int(value['correct'] and not baseline['correct']),
                regression=int(baseline['correct'] and not value['correct']),
                native_margin=baseline['margin'], margin=value['margin'], delta_margin=delta,
                candidate_sum_scores=json.dumps(value['sum_scores']),
                candidate_normalized_scores=json.dumps(value['normalized_scores']),
                native_candidate_normalized_scores=json.dumps(baseline['normalized_scores']),
                candidate_normalized_deltas=json.dumps([a-b for a,b in zip(value['normalized_scores'], baseline['normalized_scores'])]),
                elapsed_seconds_all_depths=row['elapsed_seconds']))
    write_csv(directory/'layer_intervention_results.csv', detail)
    aggregates = []
    for condition in conditions:
        match = PATTERN.match(condition)
        for k in (4, 5, 6):
            subset = [row for row in detail if row['condition']==condition and row['k']==k]
            n = len(subset)
            corrections, regressions = sum(r['correction'] for r in subset), sum(r['regression'] for r in subset)
            aggregates.append(dict(condition=condition, k=k, n=n,
                layer=int(match[3]) if match else '', branch=match[2] if match else '',
                invocation_scope=match[1] if match else 'boundary_loop5' if condition.startswith('boundary') else 'noop',
                correct=sum(r['correct'] for r in subset), native_correct=sum(r['native_correct'] for r in subset),
                accuracy=sum(r['correct'] for r in subset)/n, corrections=corrections, regressions=regressions,
                net_correct=corrections-regressions, mean_delta_margin=sum(r['delta_margin'] for r in subset)/n,
                min_delta_margin=min(r['delta_margin'] for r in subset),
                harm_group_corrections=sum(r['correction'] for r in subset if r['group']=='harm'),
                rescue_group_regressions=sum(r['regression'] for r in subset if r['group']=='rescue'),
                stable_correct_regressions=sum(r['regression'] for r in subset if r['group']=='stable_correct'),
                label='selected-case descriptive; not confirmatory'))
    write_csv(directory/'layer_aggregate_descriptive.csv', aggregates)
    # Trace files contain summaries only. Do not deserialize any tensor files.
    trace = []
    for path in sorted(directory.glob('trace_*_summary.json')):
        item = json.loads(path.read_text())
        assert item['id'] in ids and item['scope']=='official_train_discovery_only'
        for branch in item['branches']:
            if branch['loop']==5:
                trace.append(dict(id=item['id'], **branch, comparison='4_to_5',
                    interpretation='native branch contribution geometry, not causal direction evidence'))
    assert {r['id'] for r in trace} == set(ids), 'Missing native contribution trace summaries'
    write_csv(directory/'trace_contribution_alignment_4to5.csv', trace)
    # Same-position branch-change versus actual normalized loop-boundary change.
    # These cosines are descriptive; boundary RMSNorm prevents an additive
    # decomposition of the normalized final displacement into raw branch writes.
    direction_rows=[]
    for path in sorted(directory.glob('trace_*.pt')):
        import torch
        item=torch.load(path,map_location='cpu',weights_only=True)
        assert item['id'] in ids and item['implementation_sha256']==frozen['implementation_sha256']
        for candidate in item['candidates']:
            records=candidate['records']
            boundaries={r['loop']:r['normalized_native'].float() for r in records if r['kind']=='boundary'}
            displacement=boundaries[5]-boundaries[4]
            branches={(r['layer'],r['branch'],r['loop']):r for r in records if r['kind']=='branch'}
            for layer in range(24):
                for branch in ('attention','mlp'):
                    b4,b5=[branches[(layer,branch,k)]['post_original'].float() for k in (4,5)]
                    change=b5-b4
                    direction_rows.append(dict(id=item['id'],group=native[item['id']]['group'],candidate=candidate['candidate'],layer=layer,branch=branch,
                        branch_change_l2=float(change.norm()),boundary_change_l2=float(displacement.norm()),
                        branch_change_to_boundary_cosine=float(torch.nn.functional.cosine_similarity(change,displacement,dim=-1).mean()),
                        branch_write5_to_boundary_cosine=float(torch.nn.functional.cosine_similarity(b5,displacement,dim=-1).mean()),
                        interpretation='Descriptive aligned prediction positions; not additive attribution through RMSNorm'))
    if direction_rows:write_csv(directory/'branch_boundary_directional_alignment.csv',direction_rows)
    # Explicit search accounting; all-loop reruns do not add physical modules.
    physical_modules = {(r['layer'],r['branch']) for r in aggregates if r['branch']}
    loop5 = [r for r in aggregates if r['k']==5 and r['invocation_scope']=='loop5']
    ordering=lambda r:(-r['net_correct'],-r['mean_delta_margin'],r['regressions'],r['layer'],r['branch'])
    global_map = {(r['layer'],r['branch']):r for r in aggregates
                  if r['k']==5 and r['invocation_scope']=='all_loops'}
    boundaries = [r for r in aggregates if r['k']==5 and r['invocation_scope']=='boundary_loop5']
    best_boundary = max(boundaries, key=lambda r:(r['correct'],r['mean_delta_margin'])) if boundaries else None
    def local_qualified(row):
        positive=row['net_correct']>0 and row['stable_correct_regressions']==0 and row['rescue_group_regressions']==0
        beyond=best_boundary is not None and (row['correct']>best_boundary['correct'] or
            (row['correct']==best_boundary['correct'] and row['mean_delta_margin']>best_boundary['mean_delta_margin']+.001))
        return positive and beyond
    qualified_pool=[r for r in loop5 if local_qualified(r)]
    # Apply the declared scientific gate BEFORE truncating to two modules;
    # descriptive winners that damage successes must not hide a qualified lead.
    ranked=sorted(qualified_pool if qualified_pool else loop5,key=ordering)[:2]
    nominees = []
    for index, row in enumerate(ranked, 1):
        shared = global_map.get((row['layer'],row['branch']))
        positive = row['net_correct']>0 and row['stable_correct_regressions']==0 and row['rescue_group_regressions']==0
        beyond = best_boundary is not None and (row['correct']>best_boundary['correct'] or
            (row['correct']==best_boundary['correct'] and row['mean_delta_margin']>best_boundary['mean_delta_margin']+.001))
        shared_positive = shared is not None and shared['net_correct']>0 and shared['stable_correct_regressions']==0 and shared['rescue_group_regressions']==0
        status = ('eligible_for_bounded_label_blind_direction_diagnostic' if positive and beyond and shared_positive
                  else 'localization_only_requires_shared_invocation_check' if positive and beyond and shared is None
                  else 'no_current_justification_for_direction_search')
        nominees.append(dict(rank=index, **row, shared_invocation_result=shared,
            improves_over_best_tested_boundary=bool(beyond), decision=status,
            caveat='Ranking is exploratory over selected discovery cases; does not identify H2 or a weight direction.'))
    write_csv(directory/'exploratory_top_modules.csv', [{k:v for k,v in n.items()
        if k!='shared_invocation_result'} for n in nominees])
    parity = dict(status='passed', n=len(ids), native_k4_max_abs_error=0,
        noop_delta_margin_k4_k5_k6_max=max(abs(r['delta_margin']) for r in detail if r['condition']=='noop_record_all'),
        evidence='Runner COMPLETE is emitted only after exact K4/K5/K6 candidate-sum no-op assertions; this analyzer additionally checks stored zero deltas.',
        limitation='Original source candidate-sum differences are not stored separately at K5/K6; no new independent forward recomputation.')
    save(str((directory/'NOOP_GATE_SUMMARY.json').relative_to(ROOT)),parity)
    analysis = dict(status='complete', selected_representatives=len(ids),
        physical_modules_tested=len(physical_modules), expected_full_screen_modules=48,
        locally_qualified_module_count=len(qualified_pool),
        layer_conditions_tested=sum(bool(PATTERN.match(c)) for c in conditions),
        boundary_conditions_tested=len(boundaries), no_op_gate=parity,
        best_tested_boundary=best_boundary, top_at_most_two_modules=nominees,
        h2_established=False, supervised_fitting=False,
        stop_or_continue='recording_only_no_causal_screen' if not loop5
            else 'bounded_direction_diagnostic_only' if any(n['decision'].startswith('eligible') for n in nominees)
            else 'shared_invocation_confirmation_only' if any(n['decision'].startswith('localization') for n in nominees)
            else 'stop_current_H2_direction_search',
        statistical_status='No p-values: 48-module search on selected cases is exploratory and non-independent.',
        required_for_H2=['Non-collinear direction-specific intervention beyond norm-matched scalar/random controls',
                         'Transfer on untouched train selection questions',
                         'Equivalent actual BF16 shared-weight edit with no early-loop degradation'])
    save(str((directory/'MECHANISM_ANALYSIS.json').relative_to(ROOT)), analysis)
    if not args.no_plots:
        make_plots(directory, aggregates, trace)
    write_interpretation(directory, analysis, aggregates)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


def make_plots(directory, aggregates, trace):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    for scope in ('loop5','all_loops'):
        rows = sorted([r for r in aggregates if r['k']==5 and r['invocation_scope']==scope],
                      key=lambda r:(r['layer'],r['branch']))
        if not rows:
            continue
        labels = [f"L{r['layer']} {'A' if r['branch']=='attention' else 'M'}" for r in rows]
        fig, axes = plt.subplots(2,1,figsize=(13,6),sharex=True,layout='constrained')
        axes[0].bar(range(len(rows)),[r['net_correct'] for r in rows],color='#315b88')
        axes[0].set_ylabel('Net corrected cases at K5')
        axes[1].bar(range(len(rows)),[r['mean_delta_margin'] for r in rows],color='#477e68')
        axes[1].set_ylabel('Mean margin change');axes[1].set_xticks(range(len(rows)),labels,rotation=90)
        for ax in axes:ax.axhline(0,color='black',linewidth=.6)
        fig.suptitle(f'Selected discovery cases: post-norm branch damping ({scope}); exploratory')
        fig.savefig(directory/f'branch_screen_{scope}.png',dpi=180);fig.savefig(directory/f'branch_screen_{scope}.pdf');plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4),layout='constrained')
    for condition in ['noop_record_all','boundary_loop5_step_0.25','boundary_loop5_step_0.5']:
        rows=sorted([r for r in aggregates if r['condition']==condition],key=lambda r:r['k'])
        if rows:ax.plot([r['k'] for r in rows],[r['accuracy'] for r in rows],marker='o',label=condition.replace('boundary_loop5_',''))
    ax.set(xlabel='Loop depth',ylabel='Accuracy on selected discovery cases',xticks=[4,5,6],ylim=(0,1))
    ax.legend(fontsize=8);fig.savefig(directory/'boundary_downstream.png',dpi=180);fig.savefig(directory/'boundary_downstream.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for branch,color in [('attention','#315b88'),('mlp','#b16a42')]:
        layers=sorted({r['layer'] for r in trace if r['branch']==branch})
        for ax,metric,label in [(axes[0],'post_previous_cosine','Post-branch cosine: loop4 vs loop5'),
                                (axes[1],'post_change_relative_l2','Relative post-branch change')]:
            # Equal weight per question, with its candidates averaged first.
            values=[]
            for layer in layers:
                part=[r for r in trace if r['branch']==branch and r['layer']==layer]
                qids={r['id'] for r in part}
                values.append(np.mean([np.mean([r[metric] for r in part if r['id']==uid]) for uid in qids]))
            ax.plot(layers,values,marker='.',label=branch,color=color);ax.set(xlabel='Physical layer (zero based)',ylabel=label)
    axes[0].legend();fig.suptitle('Native selected-position geometry; descriptive, not causal attribution')
    fig.savefig(directory/'native_branch_alignment.png',dpi=180);fig.savefig(directory/'native_branch_alignment.pdf');plt.close(fig)


def write_interpretation(directory, analysis, aggregates):
    lines=['# Discovery layer-screen mechanism interpretation', '',
        f"Analyzed {analysis['selected_representatives']} selected training questions and {analysis['physical_modules_tested']} physical modules. These are exploratory representative-case results, not whole-discovery or held-out estimates.", '',
        '## What the screen establishes', '',
        'Exact no-op assertions passed in the completed runner. Branch damping acts after branch RMSNorm; it measures sensitivity to reducing a branch residual contribution. It does not identify a raw projection direction or establish a weight-equivalent repair.', '',
        'Boundary controls modify loop-5 normalized feedback and therefore test downstream K6 behavior. Endpoint-only rescue and feedback rescue must be reported separately. Neither alone proves H3 or rules out H1.', '',
        '## Bounded next-step decision', '',
        f"Machine-readable decision: `{analysis['stop_or_continue']}`. H2 established: **false**.", '',
        'The screening gate requires positive net K5 corrections without representative stable-correct/rescue regressions, improvement beyond the best tested boundary control, and positive all-loop transfer before permitting a bounded label-blind direction diagnostic. Margin ties use the pre-existing 0.001 descriptive epsilon. This is a resource-allocation rule, not a hypothesis test.', '']
    for row in analysis['top_at_most_two_modules']:
        lines += [f"- Exploratory rank {row['rank']}: layer {row['layer']} {row['branch']}; net K5 corrections {row['net_correct']:+d}; mean margin change {row['mean_delta_margin']:.6f}; decision `{row['decision']}`."]
    lines += ['', '## Claims still unsupported', '',
        'The best of 48 modules can overfit eight selected cases. A high output norm, directional stability, or an isolated lesion rescue is not a shared harmful direction. Absent all-loop tests means transfer is unmeasured, not failed. A stop decision means the current screen does not justify direction search, not that H2 is mathematically impossible.', '',
        'Any continuation requires a fixed, label-blind direction specification; norm-matched scalar and random directional controls; unselected train-transfer evaluation; and actual BF16 shared-weight realization. No validation or test results belong in this discovery report.', '',
        '## Reporting checklist', '',
        '- Contrast native aggregate depth results from the separate discovery report with these selected-case results.',
        '- Report every screened module and the number of intervention conditions, including negative cases.',
        '- State whether K6 feedback gains survive and whether early-loop behavior degrades under all-loop intervention.',
        '- Preserve the failed cached-BF16 readiness gate; all accepted scores here use native whole-candidate forwards.',
        '- If direction-specific evidence remains absent, report H1/H3-compatible sensitivity or unresolved mechanism rather than inventing an edit.','']
    (directory/'MECHANISM_INTERPRETATION.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':
    main()
