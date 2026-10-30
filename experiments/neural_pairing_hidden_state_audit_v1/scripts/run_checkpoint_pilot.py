"""Guarded BF16 smoke/trace pilot, never a substitute for full task evaluation.

Example (server, after server tiny checks):
 python scripts/run_checkpoint_pilot.py --upstream upstream --models E:/Codex_Research/neural_pairing_hidden_state_audit_v1/models --out pilot_01
Outputs are exclusive; existing experimental runs are not overwritten.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--models', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / 'logs/TINY_TESTS_EXIT.json').read_text())
    assert receipt['exit_code'] == 0, 'server correctness gate has not passed'
    assert len(receipt['files']) == 5
    assert receipt['upstream_commit'] == '7e7fcc401a8243f94bca94070db0993d64277874'
    models = args.models.resolve()
    # Fail before importing/loading GPU weights if staging is incomplete or altered.
    config = json.loads((root / 'configs/model_transfer.json').read_text())
    for item in config['files']:
        p = models / item['relative_path']
        assert p.stat().st_size == item['bytes'], str(p)
        h = hashlib.sha256() if item.get('sha256') else hashlib.sha1(('blob ' + str(item['bytes']) + '\0').encode())
        with p.open('rb') as f:
            for block in iter(lambda: f.read(16 << 20), b''):
                h.update(block)
        assert h.hexdigest() == (item.get('sha256') or item['git_blob']), str(p)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(NL_DEVICE='cuda:0', NL_QUILL=str(models / 'Quill-Qwen3-4B'),
        NL_EMBER=str(models / 'Ember-AFM-4.5B'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false')
    os.environ.pop('NL_TINY', None)
    sys.path.insert(0, str(args.upstream.resolve() / 'pairing'))
    import torch
    import link
    import session as S
    from evaluate import load_pair
    from causal_trace import CausalTrace
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.manual_seed(2021)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    log = dict(evidence_type='fresh_checkpoint_smoke_only', models=config['models'],
               started_unix=start, gpu=torch.cuda.get_device_name(0), gpu_worker_seconds=None,
               status='running', formal_baseline=False)
    (out / 'receipt.json').write_text(json.dumps(log, indent=2))
    try:
        pair, toks = load_pair(str(models / 'Pairing-final-nl33w'))
        pair.bridge.requires_grad_(False)
        for m in pair.models.values():
            m.eval().requires_grad_(False)
        assert not any(p.requires_grad for m in [pair.bridge, *pair.models.values()] for p in m.parameters())
        assert all(next(m.parameters()).dtype == torch.bfloat16 for m in pair.models.values())
        episodes = [dict(game='relay_word', mode='duplex',
            system={x: 'You are ' + x + '. A private neural channel connects you to your partner; you cannot read its text.' for x in link.SIDES},
            task={'quill': 'Read the private word through the channel and write it.',
                  'ember': 'The private word is "' + word + '". Out loud, say ready without revealing the word.'},
            reply={'quill': ('The word is ' + word + '.', 1.), 'ember': ('Ready.', 0.)},
            meta=dict(receiver='quill', sender='ember', value=word)) for word in ['anchor', 'velvet']]
        (out / 'smoke_episodes.json').write_text(json.dumps(episodes, indent=2))
        built = [S.build(ep, toks) for ep in episodes]
        b = S.collate(built, {x: toks[x].pad_token_id for x in link.SIDES})
        positions = {x: (b[x]['att'].sum(1) - 1)[:, None] for x in link.SIDES}
        # Predict first private payload token, before that token has been read by the receiver.
        for i, ep in enumerate(episodes):
            js = S.payload_labels(ep, toks)['quill']
            assert js
            positions['quill'][i, 0] = built[i]['quill']['labels'][js[0]][0]
        results, traces = {}, {}
        with torch.inference_mode():
            cut, _ = pair.joint(b, mode='zero')
            native_errors = {}
            for x in link.SIDES:
                native = link.inner(pair.models[x])[0](input_ids=b[x]['ids'], attention_mask=b[x]['att'],
                    position_ids=b[x]['pos'], use_cache=False).last_hidden_state
                error = float((native.float() - cut[x].float()).abs().max())
                native_errors[x] = error
                torch.testing.assert_close(native, cut[x], rtol=0, atol=0)
            reference, _ = pair.joint(b)
            for name, kw in [('link', {}), ('cut', dict(keep_receivers=[])),
                ('ember_to_quill_only', dict(keep_receivers=['quill'])),
                ('quill_to_ember_only', dict(keep_receivers=['ember']))]:
                with CausalTrace(pair, positions, **kw) as tr:
                    last, _ = pair.joint(b)
                if name == 'link':
                    for x in link.SIDES:
                        torch.testing.assert_close(reference[x], last[x], rtol=0, atol=0)
                if name == 'cut':
                    for x in link.SIDES:
                        torch.testing.assert_close(cut[x], last[x], rtol=0, atol=0)
                results[name] = {}
                for x in link.SIDES:
                    state = tr.select(x, last[x])[:, 0]
                    logits = link.inner(pair.models[x])[1](state).float()
                    results[name][x] = {'argmax_token': logits.argmax(-1).tolist(),
                        'logit_max': logits.max(-1).values.tolist()}
                    torch.save(logits.cpu(), out / (name + '_' + x + '_logits.pt'))
                torch.save(tr.states, out / (name + '_states.pt'))
                traces[name] = tr.stats
            torch.cuda.synchronize()
        log.update(status='passed', native_cut_max_abs=native_errors, results=results,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            warning='Two constructed episodes, one receiving direction. No benchmark or mechanism conclusion.')
        rows = [dict(condition=c, **r) for c, rr in traces.items() for r in rr]
        with (out / 'layerwise_smoke.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    except BaseException as exc:
        log.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        log.update(finished_unix=time.time(), gpu_worker_seconds=time.time() - start)
        (out / 'receipt.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
