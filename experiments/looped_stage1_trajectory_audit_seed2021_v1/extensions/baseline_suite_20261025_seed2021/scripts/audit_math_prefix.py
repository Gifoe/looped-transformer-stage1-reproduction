"""Read-only validation-prefix integrity audit; never selects a method."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re

from common import ROOT, BASE, CFG, PROTOCOL_SHA, read, save


def generation_fingerprint():
    files = [ROOT/'scripts/common.py', ROOT/'scripts/methods.py',
             BASE/'scripts/qwen_recipe.py', ROOT/'scripts/generate_math.py']
    return hashlib.sha256(''.join(p.name+hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in files).encode()).hexdigest()


def audit(model, require_complete=False):
    expected_sha = generation_fingerprint()
    path = ROOT/f'trajectories/math_{model}_validation.jsonl'
    # The writer can be active: a single binary read fixes our snapshot, and
    # only newline-terminated records are audited. Any trailing bytes are counted.
    payload = path.read_bytes()
    boundary = payload.rfind(b'\n')+1
    prefix = payload[:boundary]
    rows = [json.loads(line) for line in prefix.splitlines() if line.strip()]
    docs = read('data/math_validation.jsonl')
    methods = CFG['math']['validation_methods']
    expected_order = [r['id']+'|'+m for r in docs for m in methods]
    assert 0 < len(rows) <= len(expected_order)
    assert [r['record_key'] for r in rows] == expected_order[:len(rows)]
    if require_complete:
        assert len(rows) == len(expected_order) and boundary == len(payload)
    golds = {r['id']: r['gold'] for r in docs}
    loading = json.loads((ROOT/f'results/model_load_{model}.json').read_text())
    config = loading['config']; layers = config['num_hidden_layers']
    vocab = config['vocab_size']
    from transformers import AutoTokenizer
    tokenizer_dir = ROOT/'sources/tokenizers'/model
    if not tokenizer_dir.exists():tokenizer_dir = BASE/'models'/model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    trace_positions = 0
    counts = {m: 0 for m in methods}
    for r in rows:
        assert r['protocol_sha256'] == PROTOCOL_SHA
        assert r['implementation_sha256'] == expected_sha
        assert r['model'] == model and r['phase'] == 'validation' and r['task'] == 'gsm8k'
        assert r['gold'] == golds[r['id']]
        method = r['method']; counts[method] += 1
        case = r['condition']
        assert case['method'] == method and case['depth'] == 4 and case['cap'] == 0
        if method == 'baseline':
            assert case['kind'] == 'baseline' and case['omega'] == 0
        else:
            kind, strength = method.split('_')
            assert case['kind'] == kind and case['omega'] == float(strength)
        assert r['cache_layers'] == 4*layers
        tokens = r['token_ids']; n = len(tokens)
        assert n == r['new_tokens'] and n <= CFG['math']['max_new_tokens']
        assert r['max_new_tokens'] == CFG['math']['max_new_tokens']
        assert r['reached_token_limit'] == (n == CFG['math']['max_new_tokens'])
        assert all(type(t) is int and 0 <= t < vocab for t in tokens)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True).split('\nQuestion:')[0].split('\nQ:')[0]
        assert decoded == r['text']
        extracted = re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)', decoded)
        prediction = extracted[-1].replace(',', '').lstrip('+') if extracted else None
        assert r['prediction'] == prediction
        assert r['correct'] == (prediction == r['gold'].replace(',', ''))
        numeric = False
        if prediction is not None:
            try:
                numeric = Decimal(prediction) == Decimal(r['gold'].replace(',', ''))
            except InvalidOperation:
                pass
        assert r['numeric_correct'] == numeric
        assert math.isfinite(r['elapsed_s']) and r['elapsed_s'] > 0
        assert r['peak_memory_bytes'] > 0
        assert r['actual_strength_min'] == r['actual_strength_max'] == case['omega']
        assert not r.get('identity_reused_from')
        trace = r['token_trajectory']
        assert r['baseline_audit_extra_heads'] == (method == 'baseline')
        if method != 'baseline':
            assert not trace
            continue
        assert len(trace) in (n, n+1)
        assert [step['position'] for step in trace] == list(range(len(trace)))
        assert [step['chosen_token_id'] for step in trace[:n]] == tokens
        if len(trace) == n+1:
            assert trace[-1]['chosen_token_id'] == tokenizer.eos_token_id
        for step in trace:
            states = step['by_depth']
            assert [s['depth'] for s in states] == [1, 2, 3, 4]
            assert step['chosen_token_id'] == states[-1]['prediction']
            for depth, state in enumerate(states):
                assert 0 <= state['prediction'] < vocab
                for key in ['confidence', 'entropy', 'probability_margin', 'logit_margin']:
                    assert math.isfinite(state[key]) and state[key] >= -1e-5
                assert state['confidence'] <= 1.00001 and state['probability_margin'] <= 1.00001
                for key in ['full_vocab_kl', 'hidden_relative_l2', 'hidden_absolute_l2', 'hidden_cosine']:
                    value = state[key]
                    if depth == 0:
                        assert value is None
                    else:
                        assert math.isfinite(value)
                        assert abs(value) <= 1.0001 if key == 'hidden_cosine' else value >= -1e-4
            trace_positions += 1
    assert generation_fingerprint() == expected_sha
    result = dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
                  model=model, audited_records=len(rows), expected_records=len(expected_order),
                  complete_validation=(len(rows) == len(expected_order) and boundary == len(payload)), method_counts=counts,
                  audited_baseline_token_positions=trace_positions,
                  snapshot_bytes=len(payload), audited_prefix_bytes=boundary,
                  ignored_unterminated_trailing_bytes=len(payload)-boundary,
                  audited_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                  protocol_sha256=PROTOCOL_SHA, generation_implementation_sha256=expected_sha,
                  scope='Validation-prefix consistency only: order, fingerprints, token/text/extraction, strict/numeric EM, depth/cache, finite scalars and baseline traces. No test outcomes or parameter selection. Not a completion certificate.')
    save(f'results/math_prefix_audit_{model}.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Ouro-1.4B', 'Ouro-2.6B', 'all'], default='all')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    for model in ['Ouro-1.4B', 'Ouro-2.6B'] if args.model == 'all' else [args.model]:
        audit(model, args.require_complete)


if __name__ == '__main__':
    main()
