"""Check the C/D intervention against saved training and per-question scoring receipts."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from audit_ab import read, signature
from paired_analysis import METRICS, compare
from select_external import assess, ordering
from task_data import load


def math_output_lengths(rows_c, rows_d):
    paired = list(zip(rows_c, rows_d, strict=True))
    strata = {
        'full': paired,
        'same_32_token_length_bin': [(c, d) for c,d in paired if c['answer_tokens']//32 == d['answer_tokens']//32],
        'both_completed_with_process': [(c,d) for c,d in paired if all(
            r['metrics']['process'] and r['metrics']['completed'] for r in (c,d))],
    }
    return {key: dict(n=len(items), C_median=median(r[0]['answer_tokens'] for r in items),
                     D_median=median(r[1]['answer_tokens'] for r in items),
                     strict=compare(*zip(*items,strict=True))['strict']) for key,items in strata.items() if items}


def screen(fits, core, seed):
    METRICS.update(('process_correct', 'premature_eos', 'token_limit', 'eos_before_16', 'repetitive'))
    roots = {arm: fits/arm/str(seed) for arm in 'CD'}
    locks = {arm: read(root/'TRAIN_LOCK.json') for arm, root in roots.items()}
    shared_fields = ('seed', 'initial_factors_sha256', 'dataset_sha256', 'depths', 'per_step', 'model_sha256',
                     'rank_weight', 'lr', 'code_sha256')
    if any(locks['C'][key] != locks['D'][key] for key in shared_fields):
        raise ValueError('C/D differ in initialization, source pools, objective, or schedule')

    by_epoch = []
    for epoch in (1, 2, 3):
        reports = {arm: read(root/f'epoch_{epoch:02d}'/'TRAIN.json') for arm, root in roots.items()}
        steps = {arm: [row for row in doc['logs'] if row['epoch'] == epoch] for arm, doc in reports.items()}
        explanation_counts = Counter()
        if any(len(schedule) != 500 for schedule in steps.values()):
            raise ValueError(f'C/D epoch {epoch} did not complete 500 updates')
        for c, d in zip(steps['C'], steps['D'], strict=True):
            if (c['step'], c['depth']) != (d['step'], d['depth']):
                raise ValueError('C/D used different loop schedules')
            view_pairs = list(zip(c['views'], d['views'], strict=True))
            if any(a[:2] != b[:2] or (a[0] != 'scienceqa' and a != b) for a, b in view_pairs):
                raise ValueError('Question IDs or non-science views differ between C and D')
            changed = [(a,b) for a,b in view_pairs if a != b]
            if len(changed) != 1:
                raise ValueError('Each D update must replace one of the two ScienceQA short views with its explanation')
            original, replacement = changed[0]
            if original[0] != 'scienceqa' or original[2] != 'short' or replacement[2] != 'explanation':
                raise ValueError('The changed view is outside the registered ScienceQA contrast')
            explanation_counts[c['depth']] += 1

        tags = {arm: f'{arm}_s{seed}_e{epoch}' for arm in roots}
        by_case = []
        for left in sorted(core.glob(f'*__{tags["C"]}__*.jsonl')):
            dataset, _, depth, mode = left.stem.split('__')
            right = core/f'{dataset}__{tags["D"]}__{depth}__{mode}.jsonl'
            for arm, path in zip('CD', (left, right), strict=True):
                doc = read(path.with_suffix('.json'))
                if doc['rows_sha256'] != signature(path) or doc['plan']['checkpoint_sha256'] != reports[arm]['checkpoint_sha256']:
                    raise ValueError('C/D result is missing a valid development receipt')
            c_rows, d_rows = load(left), load(right)
            case = dict(dataset=dataset, depth=int(depth[1:]), mode=mode, paired=compare(c_rows,d_rows))
            if mode == 'gsm_process':
                case['output_length_and_format'] = math_output_lengths(c_rows,d_rows)
            by_case.append(case)
        if len(by_case) != 24:
            raise ValueError(f'C/D epoch {epoch} lacks the full 24-condition initial screen')
        costs = {arm: {k: doc[k] for k in ('optimizer_updates','seconds','target_tokens','padded_token_loops','checkpoint_sha256')}
                 for arm, doc in reports.items()}
        selection = {arm: ordering(assess(core, tag, 'dev-small')) for arm, tag in tags.items()}
        by_epoch.append(dict(epoch=epoch, costs=costs, explained_by_loop=dict(explanation_counts),
                             selection=selection, paired_D_minus_C=by_case))
    return dict(seed=seed, shared={k: locks['C'][k] for k in shared_fields}, epochs=by_epoch,
                cohort='Fixed development screen. No test comparisons.',
                length_and_format_caveat='Strata depend on generated text and are descriptive, not controlled causality.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', type=Path, default=Path('results/fits'))
    parser.add_argument('--development', type=Path, default=Path('results/dev-small/core'))
    parser.add_argument('--seed', type=int, default=261124137)
    parser.add_argument('--out', type=Path, default=Path('results/C_D_DEV_AUDIT.json'))
    args = parser.parse_args()
    result = screen(args.fits, args.development, args.seed)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Completed C/D audit:', [(row['epoch'], row['explained_by_loop']) for row in result['epochs']], flush=True)
