"""Independent paired KV decision/control accounting on the frozen128.

Uses saved raw candidate logits, not execution or auditor metric functions.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def pred(values):
    return max(range(len(values)), key=lambda i: values[i])


def tied(values):
    return values.count(max(values)) > 1


def main():
    cohort = json.loads((ROOT / 'inputs/confirmation128/cohort.json').read_text())
    labels = json.loads((ROOT / 'inputs/confirmation128/analysis_labels.json').read_text())
    rows = []
    source = ROOT / 'results/confirmation_base_full128_v1'
    groups = Counter()
    for path in sorted(source.glob('*.results.json')):
        obj = json.loads(path.read_text())
        native_path = path.with_name(path.name.replace('.results.json', '.native.json'))
        native = json.loads(native_path.read_text())
        assert hashlib.sha256(native_path.read_bytes()).hexdigest() == obj['native_file_sha256']
        ident, split = obj['id'], obj['split']
        gold = labels[split + ':' + ident]['gold']
        k1, k2 = [native['native'][str(k)]['candidate_logits'] for k in [1, 2]]
        correct1, correct2 = pred(k1) == gold, pred(k2) == gold
        group = ('stable_correct' if correct1 else 'rescue') if correct2 else ('damage' if correct1 else 'stable_wrong')
        groups[group] += 1
        interventions = {row['condition']: row['candidate_logits'] for row in obj['rows']
                         if row['condition'] != 'identity'}
        assert set(interventions) == {'target', 'norm_only', 'unrelated_delta'}
        rows.append({'id': ident, 'split': split, 'group': group,
                     'native1_correct': correct1, 'native2_correct': correct2,
                     'native_no_tie': not (tied(k1) or tied(k2)),
                     'target_correct': pred(interventions['target']) == gold,
                     'target_tie': tied(interventions['target']),
                     'norm_correct': pred(interventions['norm_only']) == gold,
                     'unrelated_correct': pred(interventions['unrelated_delta']) == gold,
                     'native1_logits': k1, 'native2_logits': k2, 'gold': gold,
                     'intervention_logits': interventions})
    assert len(rows) == len(cohort['ids']) == 128
    assert {r['id'] for r in rows} == set(cohort['ids'])
    losses = [r for r in rows if r['group'] == 'rescue' and not r['target_correct']]
    specific = [r for r in losses if r['norm_correct'] and r['unrelated_correct']]
    strict = [r for r in specific if r['native_no_tie'] and not r['target_tie']]
    qpath = ROOT / 'results/ROOT_CONFIRMATION128_QUERY_REVIEW.json'
    query = json.loads(qpath.read_text())
    assert query['status'] == 'PASS_COMPLETE' and not query['failures']
    qs = {}
    for condition in ['q_only', 'joint_qkv']:
        cases = [r for r in query['natural_rescue_damage_cases'] if r['condition'] == condition]
        qs[condition] = {'natural_rescue_losses': len(cases),
                         'control_specific': sum(r['control_specific'] for r in cases),
                         'control_specific_no_native_or_target_tie': sum(
                             r['control_specific'] and r['no_native_or_target_tie'] for r in cases)}
    result = {'n': 128, 'scope': 'Frozen128 paired decision accounting; no new inference, '
              'operator selection, pooling, or mediated-percentage interpretation.',
              'native_groups': dict(groups), 'kv_natural_rescue_losses': len(losses),
              'kv_both_controls_preserve': len(specific),
              'kv_both_controls_preserve_and_no_native_target_ties': len(strict),
              'kv_strict_ids': [r['id'] for r in strict], 'query': qs,
              'query_audit_sha256': hashlib.sha256(qpath.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'rows': rows}
    out = ROOT / 'results/ROOT_CONFIRMATION128_DECISION_CASES.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}))


if __name__ == '__main__':
    main()
