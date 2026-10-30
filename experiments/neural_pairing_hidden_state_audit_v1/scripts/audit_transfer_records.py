"""Recompute transfer metrics from pinned original records, not fresh inference."""
from collections import defaultdict
import json
import re

import numpy as np
from audit_published import ROOT, COMMIT, source, write_csv, paired


def has(text, value):
    return re.search(r'(?<![A-Za-z0-9])' + re.escape(value.lower()) + r'(?![A-Za-z0-9])', text.lower()) is not None


def pen_scores(rec, condition):
    value, receiver = rec['meta']['value'].lower(), rec['meta']['receiver']
    words = [(r['author'], re.sub(r'[^\w]', '', r['text']).lower()) for r in condition['words']]
    first = next((x for x, w in words if w == value), None)
    return dict(wrote=any(x == receiver and w == value for x, w in words),
                ends=bool(words) and words[-1] == (receiver, value),
                holder_first=first is not None and first != receiver)


def main():
    manifest, payload_rows, generation_rows, comparisons = {}, [], [], []
    paths = dict(nl20='results/magic-nl20/results/payload_final.json',
                 nl33w='results/magic-nl33w/payload_step300.json')
    for run, path in paths.items():
        records, digest = source(path)
        manifest[path] = digest
        groups = defaultdict(list)
        for row in records:
            assert all(np.isfinite(row[c + '_ce']) for c in ['link', 'cut', 'shuffle'])
            assert row['episode'] != row['donor'], 'same-episode shuffle is not a valid content control'
            groups[(row['game'], row['receiver'])].append(row)
        for (game, receiver), rows in sorted(groups.items()):
            n, tokens = len(rows), sum(r['n_tokens'] for r in rows)
            means = {c: sum(r[c + '_ce'] for r in rows) / n for c in ['link', 'cut', 'shuffle']}
            payload_rows.append(dict(evidence_type='published_json_reanalysis', run=run, game=game,
                receiver=receiver, n_episodes=n, n_payload_tokens=tokens,
                **{c + '_ce_episode_mean': means[c] for c in means},
                **{c + '_ce_token_weighted': sum(r[c + '_ce'] * r['n_tokens'] for r in rows) / tokens for c in means},
                cut_minus_link_ce=means['cut'] - means['link'],
                shuffle_minus_link_ce=means['shuffle'] - means['link'],
                original_record=path, upstream_commit=COMMIT))
    path = 'results/magic-nl33w/results/interleaved.json'
    records, digest = source(path)
    manifest[path] = digest
    for game, eps in records['episodes'].items():
        for ep in eps:
            for condition in ep['conditions'].values():
                if game == 'relay_word':
                    expected = bool(has(condition['text'][ep['meta']['receiver']], ep['meta']['value']))
                    assert expected == bool(condition['ok'])
                elif game == 'pen_secret':
                    recomputed = pen_scores(ep, condition)
                    assert all(bool(condition[k]) == v for k, v in recomputed.items())
                else:
                    raise ValueError('unknown game: ' + game)
        for receiver in ['quill', 'ember']:
            group = [ep for ep in eps if ep['meta']['receiver'] == receiver]
            for metric in (['ok'] if game == 'relay_word' else ['wrote', 'ends', 'holder_first', 'clean_ends']):
                def value(ep, c):
                    v = ep['conditions'][c]
                    return bool(v['ends'] and not v['holder_first']) if metric == 'clean_ends' else bool(v[metric])
                for cond in ['link', 'cut']:
                    correct = sum(value(ep, cond) for ep in group)
                    generation_rows.append(dict(evidence_type='published_json_reanalysis', run='nl33w',
                        backend=records['backend'], game=game, receiver=receiver, condition=cond,
                        metric=metric, n=len(group), successes=correct, rate=correct / len(group),
                        interpretation='contains whole word' if game == 'relay_word' else
                        'receiver authored final secret word; no earlier holder mention' if metric == 'clean_ends' else metric,
                        original_record=path, upstream_commit=COMMIT))
                comparisons.append(dict(evidence_type='published_json_reanalysis', run='nl33w', game=game,
                    receiver=receiver, metric=metric,
                    **paired([value(ep, 'link') for ep in group], [value(ep, 'cut') for ep in group])))
    write_csv(ROOT / 'published_payload_results.csv', payload_rows)
    write_csv(ROOT / 'published_generation_results.csv', generation_rows)
    write_csv(ROOT / 'published_transfer_comparisons.csv', comparisons)
    receipt = dict(evidence_type='published_json_reanalysis_not_fresh_inference', source_sha256=manifest,
        payload_summary_rows=len(payload_rows), generation_summary_rows=len(generation_rows),
        saved_generation_scores_recomputed=True,
        checkpoint_identity_note='Published bridge.pt hash and released safe tensor hash are different formats; equivalence of all parameters is not independently established.',
        limitations=['No randomized communicated-state records in these selected files.',
                     'Role alignment and raw teacher-forcing payload positions are not reconstructible from aggregate payload rows alone.',
                     'Episode/token weighted CE differ; both reported explicitly.',
                     'Contains-word accuracy is not full-response exact match.'])
    (ROOT / 'logs/PUBLISHED_TRANSFER_AUDIT.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
