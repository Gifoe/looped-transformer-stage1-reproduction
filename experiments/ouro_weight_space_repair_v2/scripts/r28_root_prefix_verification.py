"""Independent CPU audit of the saved known-prefix task graph; no effects/model."""
from pathlib import Path
import collections
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    from tokenizers import Tokenizer

    manifest = ROOT / 'configs/r28_TRUE_PREFIX_TASK_INVENTORY.json'
    x = json.loads(manifest.read_text(encoding='utf-8'))
    assert x['status'] == 'passed_CPU_prefix_admission' and not x['failure_counterexamples']
    tp = ROOT / 'artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json'
    assert hashlib.sha256(tp.read_bytes()).hexdigest() == x['metadata']['tokenizer_sha256']
    tok = Tokenizer.from_file(str(tp))
    memory = json.loads((ROOT / 'configs/r25_MEMORY_MANIFEST.json').read_text())
    source = [(e['source_token_position'], e['value_id']) for e in memory['entries']]
    tasks = {t['key']: t for t in x['tasks']}
    assert len(tasks) == len(x['tasks']) == 2366
    arrays = {}
    counts = collections.defaultdict(lambda: [0, 0, 0])
    for key, t in tasks.items():
        p = x['prefix_pool'][t['prefix_ref']]
        ids = p['ids'] + t['consumed_ids']
        assert digest(p['ids']) == t['prefix_ref'] and len(p['ids']) == p['length']
        assert len(ids) == t['length'] and digest(ids) == t['input_ids_sha256']
        assert t['read_position'] == len(ids)-1 and t['head_shape'] == [1, 1, 2048]
        assert t['independent_consumed_encoding_exact']
        text = tok.decode(ids, skip_special_tokens=False)
        assert tok.encode(text, add_special_tokens=False).ids == ids
        identity = dict(input_ids=ids, context=t['context'], mode=t['mode'], depth=t['depth'],
                        checkpoint_revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1',
                        backend='sdpa', dtype='BF16', primitive='true_prefix_single_position_head',
                        source_map_sha256=t['source_map_sha256'])
        assert digest(identity) == key
        assert t['depth'] == (5 if t['context'] == 'Q' else 4)
        assert (t['source_map_sha256'] is not None) == (t['mode'] == 'answer_block_CF4')
        if t['context'] != 'N0':
            assert all(j < len(ids) and ids[j] == v for j, v in source)
        arrays[key] = ids
        row = counts[t['context'] + '_' + t['mode']]
        row[0] += 1
        row[1] += len(ids)
        row[2] += len(ids) * t['depth']
    used = collections.Counter()
    chains = collections.defaultdict(list)
    consumers = collections.defaultdict(set)
    for u in x['usages']:
        chain = chains[(u['id'], u['candidate'])]
        chain.append(u)
        q = tasks[u['tasks']['Q']]
        assert q['context'] == 'Q' and len(q['consumed_ids']) == u['Q_position']
        consumed_text = tok.decode(q['consumed_ids'], skip_special_tokens=False)
        for name, key in u['tasks'].items():
            t = tasks[key]
            used[key] += 1
            if name != 'Q':
                assert tok.decode(t['consumed_ids'], skip_special_tokens=False) == consumed_text
                assert t['context'] == name.split('_')[0]
        for name in ('N1', 'N2'):
            assert arrays[u['tasks'][name + '_native']] == arrays[u['tasks'][name + '_CF']]
        refkeys = {k: v for k, v in u['tasks'].items() if k != 'Q'}
        assert digest(refkeys) == u['stream_group']
        consumers[u['stream_group']].add(u['tasks']['Q'])
    assert len(x['usages']) == 548 and len(chains) == 96 and set(used) == set(tasks)
    assert len({qid for qid, _ in chains}) == 24
    for key, chain in chains.items():
        chain.sort(key=lambda u: u['Q_position'])
        assert [u['Q_position'] for u in chain] == list(range(len(chain)))
        previous = []
        for j, u in enumerate(chain):
            past = tasks[u['tasks']['Q']]['consumed_ids']
            assert len(past) == j and (j == 0 or past[:-1] == previous)
            previous = past
    assert len(consumers) == len(x['stream_groups']) == 389
    for group in x['stream_groups']:
        key = digest(group['reference_tasks'])
        assert set(group['Q_consumers']) == consumers[key]
        assert len(group['Q_consumers']) == len(set(group['Q_consumers']))
    total_tokens = sum(len(v) for v in arrays.values())
    total_depth = sum(len(arrays[k]) * t['depth'] for k, t in tasks.items())
    assert total_tokens == 2186066 and total_depth == 9216897
    assert counts['Q_native'][0] == 421
    assert all(counts[n][0] == 389 for n in counts if n != 'Q_native')
    assert sum(row['Q_target_positions'] for row in x['candidate_records']) == 548
    for row in x['candidate_records']:
        for context in ('N0', 'N1', 'N2'):
            assert row['reference_target_positions'][context] == row['Q_target_positions']
            assert row['reference_position_maps'][context] == list(range(row['Q_target_positions']))
    out = dict(status='passed_independent_saved_CPU_prefix_graph_review', tasks=len(tasks),
               uses=len(x['usages']), candidates=len(chains), questions=24, reference_groups=len(consumers),
               input_tokens=total_tokens, depth_weighted_tokens=total_depth,
               context_counts_tokens_depth=dict(counts), all_source_positions_verified=143,
               consumed_reference_text_equal=True, all_saved_inputs_tokenizer_roundtrip_exact=True,
               native_CF_inputs_identical=True, complete_N0_target_maps_identity=True,
               all_task_hashes_and_dependencies_verified=True,
               manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
               scope='Saved graph, causal consumed-prefix chains, tokenizer and task/source/cache keys independently verified; original dataset extraction and lm-eval source parity remain supported by the inventory generator CPU receipt. No model or current gold/effects read.')
    output = ROOT / 'results/r28_ROOT_PREFIX_GRAPH_VERIFICATION.json'
    with output.open('x', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
