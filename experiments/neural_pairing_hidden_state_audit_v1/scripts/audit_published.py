"""Independent arithmetic from immutable published Git blobs; no model reproduction claim."""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('NP_UPSTREAM', str(ROOT.parents[1] / 'vendor/neural-pairing')))
GIT = os.environ.get('NP_GIT') or shutil.which('git') or r'D:\零件\Git\cmd\git.exe'
COMMIT = '7e7fcc401a8243f94bca94070db0993d64277874'


def source(rel):
    path = ROOT / 'sources/published' / rel
    if (REPO / '.git').exists():
        raw = subprocess.check_output([GIT, '-C', str(REPO), 'show', COMMIT + ':' + rel])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    else:
        raw = path.read_bytes()
        expected = json.loads((ROOT / 'configs/published_sources.json').read_text())['sha256'][rel]
        assert hashlib.sha256(raw).hexdigest() == expected, 'cached source hash mismatch: ' + rel
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    plus, minus = int((a & ~b).sum()), int((~a & b).sum())
    lo = lambda k: float(beta.ppf(.0125, k, n - k + 1)) if k else 0.
    hi = lambda k: float(beta.ppf(.9875, k + 1, n - k)) if k < n else 1.
    bootstrap = np.random.default_rng(2021).multinomial(n, [minus / n, (n - plus - minus) / n, plus / n], size=10000)
    interval = np.quantile((bootstrap[:, 2] - bootstrap[:, 0]) / n * 100, [.025, .975])
    return {'n': n, 'corrections': plus, 'degradations': minus, 'net': plus - minus,
            'delta_pp': 100 * (plus - minus) / n,
            'official_conservative_ci_low_pp': 100 * (lo(plus) - hi(minus)),
            'official_conservative_ci_high_pp': 100 * (hi(plus) - lo(minus)),
            'paired_bootstrap_ci_low_pp': float(interval[0]), 'paired_bootstrap_ci_high_pp': float(interval[1]),
            'mcnemar_p': float(binomtest(plus, plus + minus).pvalue) if plus + minus else 1.}


def main():
    audit, audit_sha = source('results/magic-nl28/bench_audited.json')
    expected = set(audit['exclude']['exact_input_overlap'])
    rebuilt = {j for group in ['train_test', 'val_test'] for i, j in audit['overlap'][group]['exact_input']['indices']}
    rebuilt.update(audit['test_repeated_input_indices'])
    assert expected == rebuilt and len(expected) == 31
    sensitivity = set(audit['exclude']['similar_question_overlap'])
    baselines, comparisons, groups, manifests = [], [], [], {}
    metadata = None
    for run in ['nl23', 'nl27', 'nl28']:
        result, digest = source('results/magic-' + run + '/bench.json')
        manifests[run] = {'sha256': digest, 'availability': 'published_per_question_correctness_only', 'source': 'results/magic-' + run + '/bench.json'}
        if run == 'nl28':
            assert digest == audit['scores_sha256']
        questions = result['questions']
        assert len(questions) == 3165
        if metadata is None:
            metadata = questions
        else:
            assert questions == metadata, 'Saved source/answer metadata must agree before sharing exclusions'
        all_correct = {name: np.asarray(v, bool) for name, v in result['per_question'].items()}
        assert all(len(v) == 3165 for v in all_correct.values())
        for name, values in all_correct.items():
            assert round(float(values.mean()), 4) == result['test']['all']['accuracy'][name]
        for exclusion_name, exclude in [('unfiltered', set()), ('exact_input_overlap', expected), ('similar_question_overlap', sensitivity)]:
            include = np.asarray([i not in exclude for i in range(len(questions))])
            if exclusion_name == 'exact_input_overlap':
                assert include.sum() == 3134
            for task in ['all', 'arc_challenge', 'mmlu']:
                mask = include & np.asarray([task == 'all' or q['src'] == task for q in questions])
                if not mask.any():
                    continue
                correct = {name: values[mask] for name, values in all_correct.items()}
                for name, values in correct.items():
                    baselines.append({'evidence_type': 'published_json_reanalysis', 'run': run, 'exclusion': exclusion_name, 'task': task, 'arm': name, 'n': len(values), 'correct': int(values.sum()), 'accuracy': float(values.mean()), 'upstream_commit': COMMIT})
                    if run == 'nl28':
                        saved = audit['rescored'][exclusion_name][task]['accuracy'][name]
                        assert abs(float(values.mean()) - saved) < 1e-12
                pairs = [(f'pair/{x}', f'pair/zero/{x}', 'same_checkpoint_communication') for x in ['quill', 'ember']]
                pairs += [(f'solo/{x}', f'base/{x}', 'solo_adaptation') for x in ['quill', 'ember']]
                pairs += [(f'pair/{x}', f'solo/{x}', 'joint_adaptation_and_communication') for x in ['quill', 'ember']]
                pairs += [(f'pair/{x}', f'pair/{control}/{x}', 'content_control') for x in ['quill', 'ember'] for control in ['shuffle', 'random']]
                pairs.append(('pair/' + result['answerer'], 'ensemble', 'published_answerer_vs_solo_probability_ensemble'))
                if 'pair/avg' in correct:
                    pairs += [('pair/avg', 'ensemble', 'probability_ensemble'), ('pair/avg', 'pair/zero/avg', 'same_checkpoint_communication')]
                for a, b, interpretation in pairs:
                    comparisons.append({'evidence_type': 'published_json_reanalysis', 'run': run, 'exclusion': exclusion_name, 'task': task, 'arm': a, 'comparator': b, 'interpretation': interpretation, **paired(correct[a], correct[b])})
            if exclusion_name != 'exact_input_overlap':
                continue
            cut_q, cut_e = all_correct['pair/zero/quill'], all_correct['pair/zero/ember']
            strata = {'both_correct': cut_q & cut_e, 'only_quill_correct': cut_q & ~cut_e, 'only_ember_correct': ~cut_q & cut_e, 'both_wrong': ~cut_q & ~cut_e}
            assert sum(int((v & include).sum()) for v in strata.values()) == 3134
            for name, stratum in strata.items():
                mask = include & stratum
                for side in ['quill', 'ember']:
                    before = all_correct['pair/zero/' + side][mask]
                    after = all_correct['pair/' + side][mask]
                    stat = paired(after, before)
                    wrong, right = int((~before).sum()), int(before.sum())
                    groups.append({'evidence_type': 'published_json_reanalysis', 'run': run, 'stratum': name, 'receiver': side, **stat, 'cut_wrong': wrong, 'cut_right': right, 'rescue_rate': stat['corrections'] / wrong if wrong else None, 'damage_rate': stat['degradations'] / right if right else None, 'partner_answer_imitation': 'not_identifiable_from_correctness_only', 'probability_change': None, 'margin_change': None})
    ordered = sorted(range(len(comparisons)), key=lambda i: comparisons[i]['mcnemar_p'])
    prev = 0.
    for rank, index in enumerate(ordered):
        prev = max(prev, min(1., comparisons[index]['mcnemar_p'] * (len(ordered) - rank)))
        comparisons[index]['secondary_global_holm_p'] = prev
    write_csv(ROOT / 'baseline_results.csv', baselines)
    write_csv(ROOT / 'disagreement_analysis.csv', groups)
    write_csv(ROOT / 'published_paired_comparisons.csv', comparisons)
    summary = {'evidence_type': 'published_json_reanalysis_not_independent_checkpoint_reproduction', 'upstream_commit': COMMIT, 'source_files': manifests, 'audit_source_sha256': audit_sha, 'exact_excluded_count': len(expected), 'exclusion_components': {'train_test_normalized_exact': len({j for i, j in audit['overlap']['train_test']['exact_input']['indices']}), 'val_test_normalized_exact': len({j for i, j in audit['overlap']['val_test']['exact_input']['indices']}), 'within_test_repeated_inputs': len(audit['test_repeated_input_indices'])}, 'filtered_n': 3134, 'same_saved_question_metadata_across_runs': True, 'raw_text_overlap_independently_rebuilt': False, 'saved_overlap_index_union_rebuilt': True, 'all_saved_unfiltered_accuracies_match': True, 'nl28_saved_audited_accuracies_match': True, 'nl28_original_score_hash_matches': True, 'limitations': ['Question identities in the released score JSON are only source and answer metadata; exclusion reuse assumes the saved ordering, as the authors note.', 'Correctness vectors cannot identify wrong-answer imitation, probability changes or answer margins.', 'Intervals condition on these training runs and do not cover training-seed or shared-passage variation.'], 'headline_rows': [r for r in baselines if r['run'] == 'nl23' and r['exclusion'] == 'exact_input_overlap' and r['task'] == 'all' and r['arm'] in ['pair/quill', 'solo/quill', 'ensemble']], 'baseline_rows': len(baselines), 'disagreement_rows': len(groups), 'paired_comparison_rows': len(comparisons)}
    (ROOT / 'logs/PUBLISHED_AUDIT.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
