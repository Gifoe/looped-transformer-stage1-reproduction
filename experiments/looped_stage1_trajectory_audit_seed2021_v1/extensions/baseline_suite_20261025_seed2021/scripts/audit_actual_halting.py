"""CPU checks of complete native-exit coverage against frozen MC evidence."""
import argparse
import hashlib
import json
import math
import numpy as np
from common import ROOT, BASE, CFG, PROTOCOL_SHA, read, save
from policies import exits, exit_index

CATEGORIES = ['arc_challenge', 'openbookqa', 'mmlu']
METRICS = ['full_vocab_kl', 'hidden_relative_l2', 'native_gate_cdf']
METHODS = ['fixed4', 'fixed2', *METRICS]

def category(row):
    return 'mmlu' if row['task'].startswith('mmlu_') else row['task']

def producer_fingerprint():
    files = [ROOT/'scripts/common.py', ROOT/'scripts/methods.py',
             BASE/'scripts/qwen_recipe.py', ROOT/'scripts/actual_halting.py']
    return hashlib.sha256(''.join(p.name+hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in files).encode()).hexdigest()

def selected_settings(selection):
    settings = {}
    targets = [CFG['halting']['actual_exit_validation_target'],
               *CFG['actual_exit_fallback_validation_targets']]
    for metric in METRICS:
        settings[metric] = {}
        for task in CATEGORIES:
            available = selection[metric][task]
            feasible = [target for target in targets if str(target) in available]
            assert feasible, (metric, task, 'No frozen validation setting')
            target = feasible[0]
            settings[metric][task] = dict(target=target, threshold=available[str(target)])
    return settings

def selected_documents(documents):
    selected = [r for task in CATEGORIES
                for r in [d for d in documents if category(d) == task][:10]]
    assert len(selected) == CFG['halting']['actual_latency_questions'] == 30
    assert len({r['id'] for r in selected}) == len(selected)
    return selected

def validate_records(model, measured, parity, documents, offline, selection,
                     stored_settings, layers):
    docs = {r['id']: r for r in selected_documents(documents)}
    settings = selected_settings(selection)
    assert settings == stored_settings
    repeats = CFG['halting']['actual_latency_repeats']
    assert repeats == 5
    expected_keys = {f'{uid}|{method}|{repeat}' for uid in docs
                     for method in METHODS for repeat in range(repeats)}
    assert len(measured) == len(expected_keys) == 750
    keys = [r['record_key'] for r in measured]
    assert len(set(keys)) == len(keys) and set(keys) == expected_keys
    fingerprint = producer_fingerprint()
    def expected(uid, method):
        if method.startswith('fixed'):
            return int(method[-1]), None
        threshold = settings[method][category(docs[uid])]['threshold']
        return exit_index(offline[uid]['trajectory'], method, threshold)+1, threshold

    assert parity['passed'] is True
    cases = parity['cases']
    assert len(cases) == 150
    case_keys = [(c['id'], c['method']) for c in cases]
    assert len(set(case_keys)) == len(case_keys)
    assert set(case_keys) == {(uid, method) for uid in docs for method in METHODS}
    for case in cases:
        uid, method = case['id'], case['method']
        depth, _ = expected(uid, method)
        assert case['passed'] is True and case['depth'] == depth
        assert case['actual_decoder_layer_calls'] == len(docs[uid]['request_pairs'])*layers*depth
        assert math.isfinite(case['max_score_error']) and 0 <= case['max_score_error'] < .02

    max_error = 0.
    for row in measured:
        uid, method, repeat = row['id'], row['method'], row['repeat']
        assert uid in docs and method in METHODS and type(repeat) is int
        assert row['record_key'] == f'{uid}|{method}|{repeat}'
        assert row['model'] == model and row['task'] == docs[uid]['task']
        assert row['protocol_sha256'] == PROTOCOL_SHA and row['implementation_sha256'] == fingerprint
        depth, threshold = expected(uid, method)
        assert row['depth'] == depth and row['threshold'] == threshold
        assert row['expected_decoder_layer_calls'] == len(docs[uid]['request_pairs'])*layers*depth
        # Layer hooks are intentionally absent during timed repeats; their
        # complete150-case measurement is validated above rather than invented.
        assert row['actual_decoder_layer_calls'] is None
        assert math.isfinite(row['elapsed_s']) and row['elapsed_s'] > 0
        assert type(row['peak_memory_bytes']) is int and row['peak_memory_bytes'] > 0
        raw = np.asarray(row['sum_scores'], dtype=float)
        assert raw.shape == (len(docs[uid]['choices']),) and np.isfinite(raw).all()
        assert len(row['token_counts']) == len(raw)
        assert all(type(n) is int and n > 0 for n in row['token_counts'])
        reference = offline[uid]['predictions'][f'r{depth}']
        error = float(np.max(np.abs(raw-np.asarray(reference['sum_scores']))))
        max_error = max(max_error, error)
        assert error < .02
        assert math.isclose(row['measured_repeat_max_score_error'], error, rel_tol=1e-9, abs_tol=1e-12)
        assert row['measured_repeat_quality_matches_offline'] is True
        prediction = row['result']
        assert np.array_equal(raw, np.asarray(prediction['sum_scores']))
        normalized = raw/np.asarray(docs[uid]['choice_char_lengths'])
        assert np.allclose(normalized, prediction['normalized_scores'], rtol=0, atol=1e-12)
        raw_argmax, normalized_argmax = int(raw.argmax()), int(normalized.argmax())
        choice = raw_argmax if docs[uid]['primary_metric'] == 'acc' else normalized_argmax
        assert prediction['primary_metric'] == docs[uid]['primary_metric']
        assert prediction['prediction'] == choice == reference['prediction']
        assert prediction['raw_prediction'] == raw_argmax == reference['raw_prediction']
        assert prediction['normalized_prediction'] == normalized_argmax == reference['normalized_prediction']
        assert prediction['correct'] == (choice == docs[uid]['gold'])
        assert math.isclose(prediction['gold_nll_per_token'], -raw[docs[uid]['gold']]/row['token_counts'][docs[uid]['gold']], rel_tol=1e-12)
        history = row['history']
        if method.startswith('fixed'):
            assert history == []
        else:
            assert [s['depth'] for s in history] == list(range(1, depth+1))
            assert all(s['metric'] == method for s in history)
            for position, state in enumerate(history):
                value = state['value']
                if position == 0 and method != 'native_gate_cdf':
                    assert value is None
                else:
                    assert value is not None and math.isfinite(value)
                if position < depth-1:
                    assert not exits(value, method, threshold)
            if depth < 4:
                assert exits(history[-1]['value'], method, threshold)
    return dict(passed=True, model=model, questions=len(docs), methods=METHODS,
                repetitions=repeats, timed_records=len(measured),
                layer_count_parity_cases=len(cases), maximum_recomputed_score_error=max_error,
                producer_implementation_sha256=fingerprint,
                scope='Complete frozen ID/method/repeat coverage, validation-only settings, source identity, actual parity-layer counts, offline exit/scores/quality and causal histories. Timings cover pretokenized no-cache MC scoring only.')

def audit_model(model):
    offline = {r['id']: r for r in read(f'trajectories/{model}_test.jsonl')}
    result = validate_records(model, read(f'trajectories/actual_halting_{model}.jsonl'),
                json.loads((ROOT/f'results/actual_halting_parity_{model}.json').read_text()),
                read('data/ouro_test.jsonl'), offline,
                json.loads((ROOT/'configs/halting_selected.json').read_text())[model],
                json.loads((ROOT/f'configs/actual_halting_{model}.json').read_text()),
                json.loads((ROOT/f'results/model_load_{model}.json').read_text())['config']['num_hidden_layers'])
    save(f'results/actual_halting_independent_audit_{model}.json', result)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Ouro-1.4B', 'Ouro-2.6B', 'all'], default='all')
    args = parser.parse_args()
    for name in ['Ouro-1.4B', 'Ouro-2.6B'] if args.model == 'all' else [args.model]:
        print(json.dumps(audit_model(name)), flush=True)
