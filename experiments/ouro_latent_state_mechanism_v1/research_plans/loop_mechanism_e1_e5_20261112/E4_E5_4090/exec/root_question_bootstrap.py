"""Descriptive question-resampling uncertainty for completed, audited R runs.

This adds no interventions, discovery decisions, seed replication, or thresholds.
The same resampled question indices are used for every paired contrast in a run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audits', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = {
        'method': 'paired empirical question bootstrap, percentile 95% intervals',
        'seed': 20261112,
        'resamples': 10000,
        'scope': 'All questions within each named cohort; no pooling between cohorts. '
                 'Question-resampling variability is not seed replication, globally blind '
                 'confirmation, or multiplicity-corrected causal discovery. No threshold '
                 'determines disclosure or further experiment selection.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy_version': np.__version__,
        'runs': [],
    }
    for path in args.audits:
        audit = json.loads(path.read_text(encoding='utf8'))
        assert audit['status'] == 'PASS_COMPLETE' and not audit['failures']
        rows = sorted(audit['contrast_rows'], key=lambda row: (row['split'], row['id']))
        assert len(rows) == audit['n_expected'] == audit['n_observed']
        assert len({(row['split'], row['id']) for row in rows}) == len(rows)
        # Shared resampling also preserves the within-question relation of contrasts.
        indices = np.random.default_rng(20261112).integers(0, len(rows), (10000, len(rows)))
        names = [key for key in rows[0] if key not in {'id', 'split', 'group'}]
        estimates = []
        for name in names:
            for metric in rows[0][name]:
                values = np.array([row[name][metric]['value'] for row in rows], dtype=np.float64)
                assert np.isfinite(values).all()
                means = values[indices].mean(axis=1)
                lower, upper = np.quantile(means, [0.025, 0.975])
                estimates.append({'contrast': name, 'metric': metric,
                                  'mean': float(values.mean()),
                                  'percentile95': [float(lower), float(upper)]})
        output['runs'].append({
            'audit_path': str(path),
            'audit_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'receiver_loop': audit.get('receiver_loop', 2),
            'n': len(rows),
            'ordered_ids_sha256': hashlib.sha256(json.dumps(
                [[row['split'], row['id']] for row in rows],
                separators=(',', ':'), ensure_ascii=False).encode('utf8')).hexdigest(),
            'estimates': estimates,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf8')
    print(json.dumps({'output': str(args.output), 'runs': len(output['runs']),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
