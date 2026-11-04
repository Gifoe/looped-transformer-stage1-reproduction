"""Post-run independent pooled weighted-U and cluster-bootstrap verification."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['V', 'C_energy', 'C_margin', 'C_log_tokens', 'C_neg_log_tokens']


def read(p):
    return json.loads((ROOT / p).read_text(encoding='utf8'))


def rows(p):
    return [json.loads(x) for x in (ROOT / p).read_text(encoding='utf8').splitlines()]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def independent_statistics(pairs):
    arrays, owners, group_counts = [], [], []
    for role in ['rescue', 'damage']:
        ids = sorted({p['id'] for p in pairs if p['role'] == role})
        groups = [[p['delta'] for p in pairs if p['id'] == q and p['role'] == role] for q in ids]
        arrays.append(np.asarray([x for g in groups for x in g]))
        owners.append(np.asarray([i for i, g in enumerate(groups) for _ in g]))
        group_counts.append(np.asarray([len(g) for g in groups]))
    rp, dp = arrays
    nr, nd = map(len, group_counts)
    assert nr and nd
    # Pool individual pairs; each receives reciprocal within-question mass.
    mass_r = 1.0 / group_counts[0][owners[0]]
    mass_d = 1.0 / group_counts[1][owners[1]]
    u = [(rp[:, k, None] > dp[None, :, k]).astype(float) + .5 * (rp[:, k, None] == dp[None, :, k]) for k in range(5)]
    point = np.asarray([mass_r @ matrix @ mass_d / (nr * nd) for matrix in u])
    rng = np.random.default_rng(2021)
    draws = []
    for begin in range(0, 10000, 100):
        count = min(100, 10000 - begin)
        ir = rng.integers(nr, size=(count, nr))
        jd = rng.integers(nd, size=(count, nd))
        cr = np.asarray([np.bincount(x, minlength=nr) for x in ir])
        cd = np.asarray([np.bincount(x, minlength=nd) for x in jd])
        wr = cr[:, owners[0]] * mass_r / nr
        wd = cd[:, owners[1]] * mass_d / nd
        draws.append(np.column_stack([np.einsum('bi,ij,bj->b', wr, matrix, wd, optimize=True) for matrix in u]))
    boot = np.concatenate(draws)
    return point, np.quantile(boot, [.025, .975], axis=0), np.quantile(boot[:, :1] - boot[:, 1:], [.025, .975], axis=0)


def main():
    done, sup = read('results/r22_FEATURES_COMPLETE.json'), read('logs/r22_SUPERVISOR.json')
    analysis = read('results/r22_ANALYSIS.json')
    assert done['status'] == 'complete_gold_free_features' and done['n'] == 576
    assert sup['status'] == 'complete' and sup['exit_code'] == 0 and not sup['overrun'] and not sup.get('timed_out', False)
    assert done['features_sha256'] == sha('results/r22_features.jsonl')
    assert done['protocol_sha256'] == sha('configs/r22_TRAJECTORY_LOCK.json')
    assert done['GPU_worker_seconds'] == sup['GPU_worker_seconds'] == 0 and not done['model_loaded']
    assert done['threads'] == 2 and done['first_real_checks'][0]['first_NumPy_Torch_max_error'] <= 1e-12
    manifest = read('configs/r22_CHALLENGE_MANIFEST.json')['rows']
    features = rows('results/r22_features.jsonl')
    assert len(features) == 576 and [x['id'] for x in manifest] == [x['id'] for x in features]
    labels = {x['id']: x['gold'] for split in ['discovery', 'selection'] for x in rows(f'results/r7_{split}_development.jsonl')}
    pairs, rescued = [], set()
    degenerate = positions = 0
    for row, feat in zip(manifest, features):
        assert feat['all_candidates_prefix_ids_exact'] and all(x == 0 for x in feat['first_position_maxdiff_by_depth'])
        assert feat['state_sha256'] == row['state']['sha256']
        cs, anchor, gold = feat['candidates'], row['R6_winner'], labels[row['id']]
        assert len(cs) == row['state']['candidate_count']
        for j, c in enumerate(cs):
            assert c['candidate'] == j and c['tokens'] == row['state']['choice_token_counts'][j]
            assert 0 <= c['V'] <= 1 and c['C_energy'] <= 0
            assert all(np.isfinite(c[k]) for k in NAMES)
            degenerate += c['degenerate']
            positions += c['tokens']
        for a in row['challengers']:
            role = 'damage' if anchor == gold else 'rescue' if a == gold else 'background'
            if role == 'rescue':
                rescued.add(row['id'])
            delta = [cs[a][k] - cs[anchor][k] for k in NAMES]
            if cs[a]['tokens'] == cs[anchor]['tokens'] == 1:
                assert delta[0] == 0
            pairs.append(dict(id=row['id'], split=row['split'], role=role, delta=delta, equal_tokens=cs[a]['tokens'] == cs[anchor]['tokens'], both_single=cs[a]['tokens'] == cs[anchor]['tokens'] == 1))
    expected = {x['id'] for x in rows('results/r21_oracle.jsonl') if not x['R6_correct'] and x['discrete_reachable']}
    assert rescued == expected and len(rescued) == 54
    assert degenerate == analysis['degenerate_positions'] == 0 and positions == analysis['total_positions'] == 13417
    summaries = {}
    maximum_error = 0.
    for split in ['discovery', 'selection', 'combined']:
        sub = [p for p in pairs if split == 'combined' or p['split'] == split]
        point, ci, differences = independent_statistics(sub)
        reported = analysis['results'][split]
        for k, name in enumerate(NAMES):
            maximum_error = max(maximum_error, abs(point[k] - reported['AUC'][name]), float(np.max(np.abs(ci[:, k] - reported['CI95'][name]))))
            if k:
                maximum_error = max(maximum_error, float(np.max(np.abs(differences[:, k-1] - reported['paired_difference'][name]['CI95']))))
        coverage = {}
        for mode in ['all', 'equal_tokens', 'both_single']:
            group = [p for p in sub if mode == 'all' or p[mode]]
            coverage[mode] = {role: dict(questions=len({p['id'] for p in group if p['role'] == role}), pairs=sum(p['role'] == role for p in group)) for role in ['rescue', 'damage', 'background']}
        assert coverage == analysis['coverage'][split]
        summaries[split] = dict(AUC=dict(zip(NAMES, point.tolist())), CI95={name: ci[:, k].tolist() for k, name in enumerate(NAMES)}, coverage=coverage)
    assert maximum_error <= 1e-12
    adequate = all(analysis['results'][s]['rescue_questions'] >= 10 and analysis['results'][s]['damage_questions'] >= 10 for s in ['discovery', 'selection'])
    screens = dict(power=adequate, split_AUC=all(summaries[s]['AUC']['V'] >= .65 for s in ['discovery', 'selection']), combined_CI_lower_above_half=summaries['combined']['CI95']['V'][0] > .5, all_four_control_differences_CI_positive=all(analysis['results']['combined']['paired_difference'][k]['CI95'][0] > 0 for k in NAMES[1:]))
    assert screens == analysis['screens'] and all(screens.values()) == analysis['worth_further_research']
    receipt = dict(status='passed_independent_pooled_weighted_U_cluster_bootstrap', n=576, bootstrap_repetitions_per_split=10000, maximum_statistics_absolute_error=maximum_error, all54_rescue_IDs_verified=True, state_hash_fields_and_first_position_receipts_verified=True, all_real_state_payloads_independently_reloaded=False, feature_numerics_scope='Frozen worker audits and its first independent NumPy/Torch check; no claim of a second full state extraction.', summaries=summaries, screens=screens, worth_further_research=all(screens.values()), protocol_sha256=done['protocol_sha256'], features_sha256=done['features_sha256'], GPU_worker_seconds=0)
    with (ROOT / 'results/r22_ROOT_VERIFICATION.json').open('x', encoding='utf8') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=receipt['status'], max_error=maximum_error, combined_AUC=summaries['combined']['AUC'], combined_V_CI=summaries['combined']['CI95']['V'], screens=screens)))


if __name__ == '__main__':
    main()
