"""Post-run independent Rust-tokenizer, offset, set and admission verification."""
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads((ROOT / p).read_text(encoding='utf8'))


def lines(p):
    return [json.loads(x) for x in (ROOT / p).read_text(encoding='utf8').splitlines()]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    done, sup, admission = read('results/r24_SUPPORT_COMPLETE.json'), read('logs/r24_SUPERVISOR.json'), read('results/r24_ADMISSION.json')
    spec, manifest, inventory = read('configs/r24_MEMORY_SUPPORT_LOCK.json'), read('configs/r24_INPUT_MANIFEST.json'), read('results/r24_VALUE_INVENTORY.json')
    assert done['status'] == 'complete_gold_free_support' and done['n'] == 576
    assert sup['status'] == 'complete' and sup['exit_code'] == 0 and not sup['overrun'] and not sup.get('timed_out', False)
    assert done['protocol_sha256'] == sha('configs/r24_MEMORY_SUPPORT_LOCK.json')
    assert done['support_sha256'] == sha('results/r24_support.jsonl') and done['value_inventory_sha256'] == sha('results/r24_VALUE_INVENTORY.json')
    assert done['GPU_worker_seconds'] == sup['GPU_worker_seconds'] == 0 and not done['model_or_state_or_keys_loaded']
    tokenizer_path = 'models/Ouro-1.4B/tokenizer.json'
    if not (ROOT / tokenizer_path).exists():
        tokenizer_path = 'artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json'
    assert sha(tokenizer_path) == spec['input_sha256']['models/Ouro-1.4B/tokenizer.json']
    tok = Tokenizer.from_file(str(ROOT / tokenizer_path))
    demos = read('data/slcd_fixed_demonstrations.json')['documents']
    demos = [dict(id=d['id'], question=d['question'], answer=d['choices']['text'][d['choices']['label'].index(d['answerKey'])]) for d in demos]
    assert demos == manifest['demonstrations'] and len(demos) == 25
    prefix, fields = '', []
    for demo in demos:
        prefix += 'Question: ' + demo['question'] + '\nAnswer:'
        start = len(prefix)
        prefix += ' ' + demo['answer']
        fields.append(dict(demo_id=demo['id'], start=start, end=len(prefix)))
        prefix += '\n\n'
    assert fields == inventory['answer_char_fields']
    assert hashlib.sha256(prefix.encode()).hexdigest() == inventory['fewshot_prefix_sha256']
    records = lines('results/r24_support.jsonl')
    assert len(records) == len(manifest['rows']) == 576 and [x['id'] for x in records] == [x['id'] for x in manifest['rows']]
    baselines = {x['id']: x for s in ['discovery', 'selection'] for x in lines(f'results/r7_{s}_development.jsonl')}
    recomputed = []
    for row, record in zip(manifest['rows'], records):
        assert prefix + 'Question: ' + row['current_question'] + '\nAnswer:' == row['prompt']
        enc = tok.encode(row['prompt'], add_special_tokens=False)
        entries, boundary = [], []
        for field in fields:
            for i, (a, b) in enumerate(enc.offsets):
                if a < b and a < field['end'] and b > field['start']:
                    value = dict(value_id=enc.ids[i], demo_id=field['demo_id'], source_token_position=i, offset=[a, b])
                    if field['start'] <= a and b <= field['end']:
                        assert i > 0
                        value['key_prediction_position'] = i - 1
                        entries.append(value)
                    else:
                        boundary.append(value)
        assert entries == inventory['entries'] and boundary == inventory['boundaries']
        all_values = {e['value_id'] for e in entries}
        df = {v: len({e['demo_id'] for e in entries if e['value_id'] == v}) for v in all_values}
        specific = {v for v in all_values if df[v] <= 2 and sum(c.isalnum() for c in tok.decode([v], skip_special_tokens=False).strip()) >= 2}
        assert sorted(all_values) == inventory['S_all'] and sorted(specific) == inventory['S_specific']
        assert {str(v): n for v, n in df.items()} == inventory['demo_df']
        specific_demos = sorted({e['demo_id'] for e in entries if e['value_id'] in specific})
        assert specific_demos == inventory['specific_demo_ids']
        anchor = row['R6_winner']
        bv = baselines[row['id']]['baseline_scores']['M0_R6']
        assert anchor == max(range(len(bv)), key=lambda j: bv[j]) == record['R6_winner']
        sets = []
        for j, (context, continuation) in enumerate(row['request_pairs']):
            assert context == row['prompt'] and context.endswith(':')
            whole = tok.encode(context + continuation, add_special_tokens=False).ids
            assert whole[:len(enc.ids)] == enc.ids
            target = whole[len(enc.ids):]
            assert target and digest(target) == row['target_id_sha256'][j]
            assert len(target) == row['choice_token_counts'][j]
            sets.append(set(target))
        expected = [dict(candidate=j, target_set=sorted(t), all_intersection=sorted(t & all_values), specific_intersection=sorted(t & specific), all_difference_vs_R6=sorted((t - sets[anchor]) & all_values), specific_difference_vs_R6=sorted((t - sets[anchor]) & specific)) for j, t in enumerate(sets)]
        assert expected == record['candidates']
        assert not expected[anchor]['specific_difference_vs_R6'] and not expected[anchor]['all_difference_vs_R6']
        assert record['audit']['value_inventory_sha256'] == done['value_inventory_sha256']
        recomputed.append(dict(id=row['id'], split=row['split'], R6_winner=anchor, gold=baselines[row['id']]['gold'], candidates=expected))
    summary = {}
    for split in ['discovery', 'selection', 'combined']:
        subset = [r for r in recomputed if split == 'combined' or r['split'] == split]
        wrong = [r for r in subset if r['R6_winner'] != r['gold']]
        specific_count = sum(bool(r['candidates'][r['gold']]['specific_difference_vs_R6']) for r in wrong)
        all_count = sum(bool(r['candidates'][r['gold']]['all_difference_vs_R6']) for r in wrong)
        background = {}
        for kind in ['all_wrong_candidates', 'wrong_nonanchor_candidates']:
            pairs = [(r, c) for r in subset for c in r['candidates'] if c['candidate'] != r['gold'] and (kind == 'all_wrong_candidates' or c['candidate'] != r['R6_winner'])]
            background[kind] = dict(pairs=len(pairs), specific_covered_pairs=sum(bool(c['specific_difference_vs_R6']) for r, c in pairs), all_covered_pairs=sum(bool(c['all_difference_vs_R6']) for r, c in pairs), questions=len({r['id'] for r, c in pairs}), specific_covered_questions=len({r['id'] for r, c in pairs if c['specific_difference_vs_R6']}))
        summary[split] = dict(n=len(subset), R6_errors=len(wrong), specific_gold_vs_R6_covered=specific_count, specific_proportion=specific_count / len(wrong), all_gold_vs_R6_covered=all_count, all_proportion=all_count / len(wrong), wrong_candidate_background=background)
    assert summary == admission['summary']
    gates = dict(combined_specific_20percent=summary['combined']['specific_proportion'] >= .2, eachsplit_specific_10percent=all(summary[s]['specific_proportion'] >= .1 for s in ['discovery', 'selection']), specific_types_atleast10=len(specific) >= 10, specific_demos_atleast5=len(specific_demos) >= 5, audits=True)
    assert gates == admission['gates'] and all(gates.values()) == admission['admission_passed']
    out = dict(status='passed_independent_tokenizer_offsets_sets_admission', n=576, tokenizer_sha256=sha(tokenizer_path), native_model_or_states_or_keys_loaded=False, original_demo_values_redecoded=True, all_candidate_targets_retokenized=True, all_offsets_and_fields_recomputed=True, summary=summary, S_all_types=len(all_values), S_specific_types=len(specific), value_occurrences=len(entries), specific_demo_count=len(specific_demos), admission_passed=all(gates.values()), GPU_worker_seconds=0, protocol_sha256=done['protocol_sha256'], support_sha256=done['support_sha256'], limitation='Independent deterministic support audit, not semantic enrichment, correction reachability or achieved accuracy. No keys or query probabilities computed.')
    with (ROOT / 'results/r24_ROOT_VERIFICATION.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=out['status'], S_all=out['S_all_types'], S_specific=out['S_specific_types'], value_occurrences=len(entries), specific_demo_count=len(specific_demos), combined=summary['combined'], admission=out['admission_passed'])))


if __name__ == '__main__':
    main()
