"""Replay candidate score transfer and option permutation checks on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEPTHS = (1, 2, 3, 4)
METRICS = ('acc', 'acc_norm')
CONTROLS = ('gaussian_direction', 'cross_question', 'profile_matched', 'without_attention',
            'without_mlp', 'without_attention_and_mlp')


def rows(path):
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source]


def checksum(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def candidate_scores(row, model, depth, metric):
    raw = np.asarray(row['scores'][f'{model}_k{depth}'])
    return raw / row['characters'] if metric == 'acc_norm' else raw


def score_changes(change, base):
    deltas = [x - y for x, y in zip(change, base, strict=True)]
    return np.concatenate([delta - delta.mean() for delta in deltas])


def transfer(validation):
    result = {}
    for k in DEPTHS:
        for metric in METRICS:
            native = [candidate_scores(r, 'native_cached', k, metric) for r in validation]
            full = [candidate_scores(r, 'full', k, metric) for r in validation]
            reference = score_changes(full, native)
            for control in CONTROLS:
                modified = [candidate_scores(r, control, k, metric) for r in validation]
                comparison = score_changes(modified, native)
                result[f'{control}/k{k}/{metric}'] = dict(
                    questions=len(validation), candidates=len(comparison),
                    pearson_centered_candidate_delta=float(np.corrcoef(reference, comparison)[0, 1]),
                    same_first_argmax=sum(int(x.argmax()) == int(y.argmax())
                                          for x, y in zip(full, modified, strict=True)))
    return result


def permutation(semantic):
    result = {}
    by_id = {(r['family'], r['variant']): r for r in semantic}
    families = sorted({r['family'] for r in semantic})
    pairs = {'metadata_only': ('base', 'hidden_permutation'),
             'visible_prompt': ('visible_base', 'visible_permutation')}
    for condition, variants in pairs.items():
        for k in DEPTHS:
            for metric in METRICS:
                for model in ('native', 'full'):
                    agreement = exact = max_delta = 0
                    tie_cases = []
                    for f in families:
                        p, q = (by_id[f, v] for v in variants)
                        a, b = [candidate_scores(r, model, k, metric)[np.argsort(r['canonical_options'])]
                                for r in (p, q)]
                        agreement += int(np.array_equal(a == a.max(), b == b.max()))
                        exact += int(np.array_equal(a, b))
                        max_delta = max(max_delta, float(np.abs(a - b).max()))
                        if max((a == a.max()).sum(), (b == b.max()).sum()) > 1:
                            tie_cases.append(f)
                    result[f'{condition}/{model}/k{k}/{metric}'] = dict(
                        n=len(families), exact_canonical_score_pairs=exact,
                        canonical_top_set_agreement=agreement, max_absolute_canonical_score_delta=max_delta,
                        tied_top_family_ids=tie_cases)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    validation = args.results / 'DIRECTION_VALIDATION_SCORES.jsonl'
    semantics = args.results / 'SEMANTIC_SENSITIVITY_SCORES.jsonl'
    per_question, semantic_rows = rows(validation), rows(semantics)
    if len(per_question) != 299 or len(semantic_rows) != 72:
        raise ValueError('Expected 299 Validation rows and 72 semantic-variant rows')
    report = dict(input_sha256={p.name: checksum(p) for p in (validation, semantics)},
                  score_transfer=transfer(per_question), permutation=permutation(semantic_rows),
                  score_transfer_definition='Pearson correlation of candidate score changes vs cached native, centered within each question, pooled across candidates.',
                  exact_tie_rule='Compare sets of canonical top options. A first-index tie break can change with storage order.')
    path = args.results / 'DIRECTION_REPLAY_AUDIT.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


if __name__ == '__main__':
    main()
