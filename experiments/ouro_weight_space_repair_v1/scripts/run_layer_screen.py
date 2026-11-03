"""Discovery-only causal screening; native whole-candidate forwards, no KV cache.

Run under supervise.py for charged GPU budget. Does not read validation/test files.
Default: full recording/no-op, two boundary controls, 48 loop5 branch damp screens.
--all-loops adds 48 shared-invocation damping controls; no weight edits are fitted.
"""
import argparse
import hashlib
import json
import time

from runtime import ROOT, load_model, encode_pair, save, check_budget, outcome
from ouro_cached_scoring import OuroCachedScorer
from causal_hooks import OuroCausalHooks, BranchIntervention, BoundaryDamping


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append(path, row):
    with path.open('a', encoding='utf8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def trace_summary(records, candidate):
    """Selected continuation prediction positions only; no prefix activations."""
    import torch
    summaries = []
    previous = {}
    for record in records:
        if record['kind'] != 'branch':
            continue
        raw = record['raw_original'].float()
        post = record['post_original'].float()
        key = record['layer'], record['branch']
        row = dict(candidate=candidate, layer=record['layer'], branch=record['branch'],
                   loop=record['loop'], prediction_positions=record['positions'],
                   raw_l2=float(raw.norm()), post_l2=float(post.norm()),
                   raw_mean_token_l2=float(raw.norm(dim=-1).mean()),
                   post_mean_token_l2=float(post.norm(dim=-1).mean()))
        if key in previous:
            old = previous[key]
            delta = post-old
            row.update(post_change_l2=float(delta.norm()),
                post_change_relative_l2=float(delta.norm()/old.norm().clamp_min(1e-12)),
                post_previous_cosine=float(torch.nn.functional.cosine_similarity(post, old, dim=-1).mean()))
        previous[key] = post
        summaries.append(row)
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='layer_screen_v1')
    ap.add_argument('--max-cases', type=int, default=8)
    ap.add_argument('--all-loops', action='store_true')
    ap.add_argument('--record-only', action='store_true')
    ap.add_argument('--no-save-tensors', action='store_true')
    args = ap.parse_args()
    assert args.study.replace('_', '').isalnum()
    assert 1 <= args.max_cases <= 8
    ready = json.loads((ROOT/'results/READINESS.json').read_text())
    assert ready['status'] == 'passed'
    representative_path = ROOT/'configs/representative_cases.json'
    representative = json.loads(representative_path.read_text())
    source_study = representative['source_study']
    assert source_study.replace('_', '').isalnum()
    source_dir = ROOT/'results'/source_study
    assert (source_dir/f"COMPLETE_{representative['source_n']}.json").exists()
    # These explicit train-only inputs are the ONLY example/prediction files read.
    discovery_path = ROOT/'data/discovery.jsonl'
    discovery = {r['id']: r for r in map(json.loads, discovery_path.read_text().splitlines())}
    source_scores_path = source_dir/'scores.jsonl'
    source_scores = {r['id']: r for r in map(json.loads, source_scores_path.read_text().splitlines())}
    assert len(source_scores) == representative['source_n']
    cases = representative['cases'][:args.max_cases]
    assert cases and len({r['id'] for r in cases}) == len(cases)
    for case in cases:
        row, score = discovery[case['id']], source_scores[case['id']]
        assert '/train/' in row['id'] and score['scope'] == 'official_train_discovery_only'
        assert score['cached'] is False and score['fewshot'] == 25
        assert score['prompt_sha256'] == row['prompt_sha256']
    files = ['runtime.py', 'ouro_cached_scoring.py', 'causal_hooks.py', 'run_layer_screen.py']
    codehash = hashlib.sha256(''.join(digest(ROOT/'scripts'/name) for name in files).encode()).hexdigest()
    frozen = dict(implementation_sha256=codehash, representative_sha256=digest(representative_path),
        discovery_sha256=digest(discovery_path), native_scores_sha256=digest(source_scores_path),
        ids=[r['id'] for r in cases], all_loops=args.all_loops, record_only=args.record_only,
        save_tensors=not args.no_save_tensors, alpha=.5, boundary_steps=[.25,.5], scoring='native_uncached')
    out = ROOT/'results'/args.study
    out.mkdir(exist_ok=True)
    config_path = out/'FROZEN.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == frozen, 'Resume inputs/code changed'
    else:
        save(str(config_path.relative_to(ROOT)), frozen)
    scores_path = out/'scores.jsonl'
    completed = {}
    if scores_path.exists():
        for line in scores_path.read_text().splitlines():
            item = json.loads(line)
            key = item['id'], item['condition']
            assert key not in completed and item['implementation_sha256'] == codehash
            completed[key] = item
    started = time.time()
    tok, model = load_model()
    scorer = OuroCachedScorer(model, 6)
    import torch
    assert len(model.model.layers) == 24
    recording = [BranchIntervention(layer, branch, mode='record')
                 for layer in range(24) for branch in ('attention', 'mlp')]
    conditions = [('noop_record_all', recording, None, True)]
    if not args.record_only:
        # Diagnose trajectory effects BEFORE attempting broad localization.
        conditions += [(f'boundary_loop5_step_{step:g}', [], BoundaryDamping(5, step), False)
                       for step in (.25, .5)]
        for scope, loops in [('loop5', (5,))] + ([('all_loops', None)] if args.all_loops else []):
            conditions += [(f'damp_{scope}_{branch}_layer{layer:02d}',
                [BranchIntervention(layer, branch, mode='damp', alpha=.5, loops=loops)], None, False)
                for layer in range(24) for branch in ('attention', 'mlp')]
    expected = len(cases)*len(conditions)
    encoded = {case['id']: [encode_pair(tok, *pair) for pair in discovery[case['id']]['request_pairs']]
               for case in cases}
    # Complete one condition across all representatives before the next; this
    # preserves interpretable balanced progress if the diagnostic budget runs out.
    for case, chosen_condition in [(case, condition) for condition in conditions for case in cases]:
        row = discovery[case['id']]
        pairs = encoded[case['id']]
        for condition, branches, boundary, record in [chosen_condition]:
            if (row['id'], condition) in completed:
                continue
            # Supervising deadline must reserve enough for this bounded case.
            check_budget(90)
            began = time.time()
            scored, summaries, traces = [], [], []
            try:
                with OuroCausalHooks(model, branches, boundary) as hooks:
                    scorer.invalidate()  # acknowledge hook registration; never use cache
                    for index, (prefix, target) in enumerate(pairs):
                        check_budget(30)
                        # Whole-candidate forward: these are ALL answer prediction positions.
                        positions = list(range(len(prefix)-1, len(prefix)+len(target)-1))
                        hooks.set_capture_positions(positions if record else [])
                        single = scorer.score_uncached([(prefix, target)], loopcd=False)[0]
                        scored.append(single)
                        if record:
                            records = hooks.records
                            assert sum(r['kind']=='branch' for r in records) == 24*2*6
                            summaries += trace_summary(records, index)
                            if not args.no_save_tensors:
                                compact = []
                                for item in records:
                                    # record-only duplicates are equal; save one raw/post copy.
                                    compact.append({k:v for k,v in item.items()
                                        if k not in ('raw_treated','post_treated','normalized_treated')})
                                traces.append(dict(candidate=index, target_ids=target, records=compact))
            finally:
                scorer.invalidate()  # context removal also changes fingerprint
            methods = {f'native_k{k}': outcome([s['sum_loglikelihood'][k] for s in scored],
                       row['choice_char_lengths'], row['gold']) for k in (4,5,6)}
            baseline = source_scores[row['id']]['methods']
            if record:
                parity = {f'k{k}': max(abs(a-b) for a,b in zip(methods[f'native_k{k}']['sum_scores'],
                    baseline[f'native_k{k}']['sum_scores'])) for k in (4,5,6)}
                if any(error != 0 for error in parity.values()):
                    save(str((out/'NOOP_PARITY_FAILURE.json').relative_to(ROOT)),
                         dict(id=row['id'], error=parity, implementation_sha256=codehash))
                    raise RuntimeError('Recording-only hooks failed EXACT native-score parity')
                stem = hashlib.sha256(row['id'].encode()).hexdigest()[:16]
                save(str((out/f'trace_{stem}_summary.json').relative_to(ROOT)),
                     dict(id=row['id'], scope='official_train_discovery_only', branches=summaries))
                if traces:
                    torch.save(dict(id=row['id'], candidates=traces, implementation_sha256=codehash),
                               out/f'trace_{stem}.pt')
            # Fifth-loop interventions cannot change earlier K4 outputs.
            early_error = max(abs(a-b) for a,b in zip(methods['native_k4']['sum_scores'],
                                                      baseline['native_k4']['sum_scores']))
            if boundary is not None or (branches and all(b.loops==(5,) for b in branches)):
                assert early_error == 0, ('Earlier loop changed under loop5-only intervention', early_error)
            torch.cuda.synchronize()
            item = dict(id=row['id'], group=case['group'], condition=condition,
                scope='official_train_discovery_only', cached=False, methods=methods,
                native_k4_max_abs_error=early_error,
                delta_margin={f'k{k}': methods[f'native_k{k}']['margin']-baseline[f'native_k{k}']['margin']
                              for k in (4,5,6)},
                candidate_token_counts=[len(target) for _,target in pairs],
                elapsed_seconds=time.time()-began, implementation_sha256=codehash,
                intervention_semantics='post_branch_norm_scalar_damping' if condition.startswith('damp')
                    else 'normalized_boundary_chord_fed_to_loop6' if boundary is not None else 'exact_noop_recording')
            append(scores_path, item)
            completed[(row['id'], condition)] = item
            progress = dict(status='running', completed=len(completed), expected=expected,
                            last_id=row['id'], last_condition=condition, elapsed_seconds=time.time()-started)
            save(str((out/'PROGRESS.json').relative_to(ROOT)), progress)
            print(json.dumps(progress), flush=True)
    assert len(completed) == expected
    save(str((out/'COMPLETE.json').relative_to(ROOT)), dict(status='complete',
         representative_count=len(cases), conditions=len(conditions), completed=len(completed),
         elapsed_seconds=time.time()-started, implementation_sha256=codehash,
         interpretation='Selected-case localization only; no H2 or held-out transfer claim.'))


if __name__ == '__main__':
    main()
