"""Render the completed frozen analysis; no fitting, policy selection or new GPU work."""
from pathlib import Path
import csv
import json
import hashlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from verify_completed import ROOT, OLD, ALPHAS, KINDS, load, read_rows, independent_choose

ACTION_NAMES = ['Continue', 'Rollback', 'Partial rollback', 'Extrapolate']
COLORS = ['#4C78A8', '#E45756', '#72B7B2']


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(fig, name):
    fig.savefig(ROOT / 'figures' / (name + '.png'), dpi=200, bbox_inches='tight')
    fig.savefig(ROOT / 'figures' / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    audit = load(ROOT / 'results/INDEPENDENT_CPU_VERIFICATION.json')
    assert audit['passed']
    report = load(ROOT / 'results/VALIDATION_ANALYSIS.json')
    assert audit['analysis_sha256'] == hashlib.sha256((ROOT / 'results/VALIDATION_ANALYSIS.json').read_bytes()).hexdigest()
    rows = read_rows(ROOT / 'results/validation.jsonl')
    references = {r['id']: r for r in read_rows(OLD / 'results/validation.jsonl')}
    models = load(ROOT / 'configs/heads.json')
    fit = load(ROOT / 'results/FIT.json')
    n = len(rows)
    gold = np.asarray([r['gold'] for r in rows])
    right = np.asarray([[r['actions'][a]['prediction'] == r['gold'] for a in ALPHAS] for r in rows])
    native = np.asarray([int(np.argmax(np.asarray(references[r['id']]['views']['official']['sum_scores']['keep']) / r['choice_char_lengths'])) == r['gold'] for r in rows])
    events = []
    retained = {}
    for kind in KINDS:
        selected = np.asarray([ALPHAS.index(independent_choose(r, models[kind])[0]) for r in rows])
        chosen = right[np.arange(n), selected]
        for j, action in enumerate(ALPHAS[1:], 1):
            delta = right[:, j].astype(int) - right[:, 0].astype(int)
            is_selected = selected == j
            events.append(dict(head=kind, action=action, beneficial_total=int((delta > 0).sum()), harmful_total=int((delta < 0).sum()), beneficial_selected=int(((delta > 0) & is_selected).sum()), beneficial_not_selected=int(((delta > 0) & ~is_selected).sum()), harmful_selected=int(((delta < 0) & is_selected).sum()), harmful_not_selected=int(((delta < 0) & ~is_selected).sum()), neutral_selected=int(((delta == 0) & is_selected).sum())))
        benefits = right[:, 0] & ~native
        harms = ~right[:, 0] & native
        retained[kind] = dict(adaptive_benefits_total=int(benefits.sum()), adaptive_benefits_retained=int((benefits & chosen).sum()), adaptive_benefits_lost=int((benefits & ~chosen).sum()), adaptive_harms_total=int(harms.sum()), adaptive_harms_repaired=int((harms & chosen).sum()), adaptive_harms_remaining=int((harms & ~chosen).sum()))
    save_csv(ROOT / 'results/ACTION_ACCEPTANCE_EVENTS.csv', events)
    (ROOT / 'results/LOOPCD_CORRECTION_RETENTION.json').write_text(json.dumps({'scope': 'Secondary accounting relative to the same-question original native four-loop readout; does not change the frozen selection gate', 'heads': retained}, indent=2), encoding='utf8')
    summaries = [dict(method=k, **v) for k, v in report['summary_vs_adaptive'].items()]
    save_csv(ROOT / 'results/POLICY_SUMMARY.csv', [{**r, 'ci95_pp': json.dumps(r['ci95_pp'])} for r in summaries])
    auc_rows = []
    for action, signal in report['signals'].items():
        for kind, result in signal['heads'].items():
            auc_rows.append(dict(action=action, signal=kind, beneficial=signal['beneficial'], harmful=signal['harmful'], auc=result['auc'], ci95=json.dumps(result['ci95'])))
    save_csv(ROOT / 'results/AUC_SUMMARY.csv', auc_rows)

    (ROOT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    methods = ['adaptive', 'raw_probe', 'confidence', 'input', 'combined', 'oracle']
    labels = ['Adaptive', 'Raw input', 'Confidence head', 'Input head', 'Combined head', 'Oracle']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={'width_ratios': [1.1, 1.1, 1.3]}, constrained_layout=True)
    summary = report['summary_vs_adaptive']
    accuracy = [summary[k]['accuracy_percent'] for k in methods]
    axes[0].barh(np.arange(len(methods)), accuracy, color=['#999999', '#BBBBBB', COLORS[1], COLORS[0], COLORS[2], '#D8B365'])
    for i, k in enumerate(methods):
        axes[0].text(accuracy[i] + .3, i, str(summary[k]['correct']), va='center', fontsize=9)
    axes[0].set(yticks=np.arange(len(methods)), yticklabels=labels, xlabel='Accuracy (%)', title=f'Head-held-out development (n={n})', xlim=(0, max(accuracy) + 6))
    axes[0].invert_yaxis()
    for i, kind in enumerate(KINDS):
        result = summary[kind]
        lo, hi = result['ci95_pp']
        axes[1].errorbar(result['delta_pp'], i, xerr=np.asarray([[result['delta_pp'] - lo], [hi - result['delta_pp']]]), fmt='o', color=COLORS[i], capsize=4)
    axes[1].axvline(0, color='#777777', linewidth=1)
    axes[1].set(yticks=range(3), yticklabels=KINDS, xlabel='Accuracy change vs Adaptive (pp)', title='Paired 95% bootstrap intervals')
    axes[1].invert_yaxis()
    left = np.zeros(3)
    for j, a in enumerate(ALPHAS):
        counts = np.asarray([report['selected_action_counts'][k][a] for k in KINDS])
        axes[2].barh(range(3), counts, left=left, label=ACTION_NAMES[j], color=['#999999', '#4C78A8', '#72B7B2', '#ECA82C'][j])
        for i, count in enumerate(counts):
            if count >= n * .08:
                axes[2].text(left[i] + count / 2, i, str(count), ha='center', va='center', fontsize=8)
        left += counts
    axes[2].set(yticks=range(3), yticklabels=KINDS, xlabel='Selected questions', title='Action allocation', xlim=(0, n))
    axes[2].invert_yaxis()
    axes[2].legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, frameon=False, fontsize=8)
    export(fig, 'heldout_controller')

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), constrained_layout=True)
    for j, action in enumerate(ALPHAS[1:]):
        for i, kind in enumerate(KINDS):
            result = report['signals'][action]['heads'][kind]
            if result['auc'] is not None:
                lo, hi = result['ci95']
                axes[j].errorbar(result['auc'], i, xerr=np.asarray([[result['auc'] - lo], [hi - result['auc']]]), fmt='o', color=COLORS[i], capsize=4)
        signal = report['signals'][action]
        axes[j].axvline(.5, color='#999999', linewidth=1, linestyle='--')
        axes[j].set(xlim=(-.03, 1.03), yticks=range(3), yticklabels=KINDS, xlabel='Beneficial vs harmful AUC', title=ACTION_NAMES[j + 1] + f"\n{signal['beneficial']} beneficial, {signal['harmful']} harmful")
        axes[j].invert_yaxis()
    export(fig, 'heldout_value_ranking')

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), constrained_layout=True)
    for j, action in enumerate(ALPHAS[1:]):
        limits = [0.]
        for i, kind in enumerate(KINDS):
            bins = [b for b in report['calibration'][kind][action]['validation_bins'] if b['n']]
            x = [b['predicted_mean'] for b in bins]
            y = [b['observed_mean'] for b in bins]
            axes[j].plot(x, y, '-o', color=COLORS[i], label=kind, markersize=4)
            limits += x + y
        lo, hi = min(limits), max(limits)
        pad = max(.003, (hi - lo) * .08)
        axes[j].plot([lo - pad, hi + pad], [lo - pad, hi + pad], '--', color='#999999', linewidth=1)
        axes[j].set(xlabel='Mean predicted accuracy gain', ylabel='Mean observed accuracy gain', title=ACTION_NAMES[j + 1])
    axes[-1].legend(frameon=False)
    fig.suptitle('Validation calibration in bins frozen from training; descriptive, no bin tuning', fontsize=10)
    export(fig, 'heldout_value_calibration')

    selected = report['selected_policy']
    verdict = 'No-Go for this calibrated action family' if selected == 'adaptive' else 'Passed development screen; independent confirmation still required'
    lines = ['# Frozen-state action-value calibration', '', f'**{verdict}.**', '', f"Training used 3,200 OpenBookQA questions; the head-held-out development evaluation used {n} different questions. Both splits were historically exposed in earlier experiments. These results do not establish independent generalization. The frozen selection rule chose `{selected}`.", '', '## What the intermediate variable estimates', '', 'At recurrent depth two, the input head maps the changes in known-input token loss on two fixed probe groups into expected final accuracy change for each state action. The target is the final correctness indicator under that action minus the Adaptive correctness indicator. Continue has value zero; the policy chooses the largest positive predicted gain. It runs the remaining recurrent iterations after applying the chosen state action.', '', 'The input and confidence heads each contain nine fitted linear coefficients, including intercepts. The combined head contains fifteen. Feature normalization statistics are stored separately. Only the small heads were supervised; Ouro weights were frozen. No evaluation-label threshold or sign search was performed.', '', '## Controller outcomes', '', '| Policy | Correct | Accuracy | Corrections | Degradations | Net | Paired 95% CI (pp) |', '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for name, s in summary.items():
        lines.append(f"| {name} | {s['correct']}/{n} | {s['accuracy_percent']:.2f}% | {s['corrections']} | {s['degradations']} | {s['net']:+d} | [{s['ci95_pp'][0]:.3f}, {s['ci95_pp'][1]:.3f}] |")
    lines += ['', 'Primary admission required at least five net corrections, a positive lower paired interval and Holm-adjusted p<0.05 versus Adaptive, the equally calibrated confidence head and the training-selected best fixed action, plus a significant action-frequency-preserving shuffle comparison. All eight primary p values were adjusted together.', '', '| Primary test | Holm p |', '| --- | ---: |']
    for test, p in report['primary_holm_eight_p'].items():
        lines.append(f'| {test} | {p:.6g} |')
    lines += ['', '## Signal evidence and its limits', '', '| Action | Beneficial / harmful events | Input AUC | Confidence AUC | Combined AUC |', '| --- | --- | --- | --- | --- |']
    for j, action in enumerate(ALPHAS[1:]):
        signal = report['signals'][action]
        cells = []
        for kind in KINDS:
            s = signal['heads'][kind]
            cells.append('undefined' if s['auc'] is None else f"{s['auc']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
        lines.append(f"| {ACTION_NAMES[j+1]} | {signal['beneficial']} / {signal['harmful']} | {' | '.join(cells)} |")
    lines += ['', 'These ranking intervals and paired AUC differences are secondary diagnostics. AUC on nonzero accuracy-change events does not prove a useful acceptance policy. Calibration-bin plots, all-row squared error, and the zero/training-mean controls are in the frozen analysis JSON. `ACTION_ACCEPTANCE_EVENTS.csv` reports beneficial operations selected/rejected and harmful operations selected/avoided for every head and action; `LOOPCD_CORRECTION_RETENTION.json` accounts for original Adaptive corrections retained and harms repaired.', '', '## Cost and mechanism', '', f"The collection charged {audit['gpu_charge_seconds']/3600:.4f} GPU-worker hours within this stage. This is exhaustive counterfactual research cost, not deployment cost.", '', '| Head | Serial controller / Adaptive time |', '| --- | ---: |']
    for kind in KINDS:
        lines.append(f"| {kind} | {report['timing'][kind]['ratio']:.4f} |")
    lines += ['', 'Runtime ratios replay measured branches and include the two-loop decision probe and confidence diagnostics. They do not establish optimized serving latency. All selected branches still execute four recurrent calls; this protocol does not demonstrate early stopping or saved recurrent work. Full rollback of the second homogeneous recurrent state largely behaves like one fewer effective refinement, so rollback alone is not a novelty claim.', '', 'Terminal-only selected-action ablations and paired comparisons appear in `VALIDATION_ANALYSIS.json`. Useful state control must be distinguished from a better terminal readout. An equal-compute prompt ensemble and independent benchmarks remain prerequisites for a broad accuracy–compute or generalization claim if a candidate passes this screen.', '', '## Research decision', '']
    if selected == 'adaptive':
        lines += ['No head met the frozen development admission rule. Retain Adaptive and do not open an independent final benchmark for this candidate family. The experiment does not establish a reliable continuation/feedback variable, and the full research objective remains unachieved. Increasing head size or tuning thresholds on these outcomes would not repair the evidence.']
    else:
        lines += [f'Freeze `{selected}` and its stored coefficients before any independent confirmation. A passed development screen supports further testing; superiority on untouched tasks, equal-compute comparisons, transfer to Qwen, and scientific novelty remain unproven.']
    lines += ['', 'Training-free evidence for this state-action family was previously negative: the 400-question probe controller obtained 190 correct versus Adaptive 192. This stage tests whether bounded supervision contributes usable calibration; it does not retroactively make that rule training-free.', '', '![Held-out controllers](figures/heldout_controller.png)', '', '![Action-value ranking](figures/heldout_value_ranking.png)', '', '![Calibration](figures/heldout_value_calibration.png)', '']
    (ROOT / 'FINDINGS.md').write_text('\n'.join(lines), encoding='utf8')
    print(json.dumps({'report_written': True, 'selected_policy': selected, 'verdict': verdict, 'n': n, 'training_events': fit['training_action_events']}))


if __name__ == '__main__':
    main()
