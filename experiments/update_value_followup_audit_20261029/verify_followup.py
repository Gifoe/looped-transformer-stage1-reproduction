"""Replay stored coefficients and independently verify paired counts; no fitting."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'frozen_loop_update_value_seed2021_v1'
sys.path.insert(0, str(BASE / 'scripts'))
from analysis_core import features, ACTIONS

def load(p):
    return json.loads(p.read_text(encoding='utf8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    original = load(BASE / 'ARTIFACT_MANIFEST.json')['files']
    for rel, entry in original.items():
        assert sha(BASE / rel) == entry['sha256'], rel
    lock = load(ROOT / 'CONDITIONAL_LOCK.json')
    for rel, digest in lock['files'].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in lock['original_inputs'].items():
        assert sha(BASE / rel) == digest, rel
    with (BASE / 'results/validation.jsonl').open(encoding='utf8') as f:
        rows = [json.loads(line) for line in f]
    gold = np.array([r['gold'] for r in rows])
    pred = np.array([[np.argmax(np.asarray(r['views']['official']['sum_scores'][a]) / r['choice_char_lengths']) for a in ACTIONS] for r in rows])
    models = load(ROOT / 'CONDITIONAL_MODELS.json')
    saved = load(ROOT / 'CONDITIONAL_PREDICTIONS.json')
    result = load(ROOT / 'CONDITIONAL_RESULTS.json')
    counts = {}
    for name, model in models.items():
        kind = name.removeprefix('conditional_')
        X = np.stack([features(r, kind)[0] for r in rows])
        values = []
        for j, m in enumerate(model['actions'], 1):
            value = (X - m['mean']) / m['scale'] @ np.array(m['coef']) + m['intercept']
            values.append(np.where(pred[:, j] != pred[:, 0], value, 0))
        values = np.stack(values, axis=1)
        best = values.argmax(axis=1)
        action = np.where(values[np.arange(len(rows)), best] > model['threshold'], best + 1, 0)
        chosen = pred[np.arange(len(rows)), action]
        records = [r for r in saved if r['policy'] == name]
        assert len(records) == len(rows) == 800
        for i, r in enumerate(records):
            assert r['id'] == rows[i]['id'] and r['gold'] == gold[i]
            assert r['prediction'] == chosen[i] and r['action'] == ACTIONS[action[i]]
            np.testing.assert_allclose(r['values'], values[i], rtol=1e-10, atol=1e-10)
        right, base_right = chosen == gold, pred[:, 0] == gold
        corrections = int((right & ~base_right).sum())
        degradations = int((~right & base_right).sum())
        p = binomtest(corrections, corrections + degradations, 0.5).pvalue
        summary = result['summary'][name]
        assert summary['correct'] == int(right.sum())
        assert summary['corrections'] == corrections and summary['degradations'] == degradations
        assert summary['net'] == corrections - degradations
        assert abs(summary['mcnemar_p'] - p) < 1e-12
        counts[name] = dict(correct=int(right.sum()), corrections=corrections, degradations=degradations, mcnemar_p=p)
    assert not result['passes_numerical_development_screen']
    report = dict(original_payload_hashes_verified=len(original), conditional_locks_verified=True, validation_predictions_replayed=len(saved), paired_counts_verified=counts, final_tasks_loaded=False, additional_gpu_seconds=0)
    (ROOT / 'VERIFICATION.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
