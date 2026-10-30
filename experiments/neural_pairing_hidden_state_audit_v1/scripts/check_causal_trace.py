"""Executable infrastructure assertions; random tiny weights are not task evidence."""
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('NP_UPSTREAM', str(ROOT.parents[1] / 'vendor/neural-pairing')))
STORE = Path(os.environ.get('NP_MODEL_ROOT', str(ROOT / 'transport/model_transfer_cache')))
os.environ.update(NL_DEVICE='cpu', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
sys.path.insert(0, str(REPO / 'pairing'))

import torch
import link
from causal_trace import CausalTrace


def main():
    torch.set_num_threads(2)
    start = time.time()
    aliases = dict(quill='Quill-Qwen3-4B', ember='Ember-AFM-4.5B')
    models = {x: link.tiny(str(STORE / alias)).eval().requires_grad_(False) for x, alias in aliases.items()}
    pair = link.Pair(models, {}, (1, 2, 4), heads=2, head_dim=16).eval().requires_grad_(False)
    with torch.no_grad():
        for side in pair.bridge.values():
            side.alpha.fill_(.5)
    b, positions = {}, {}
    for x, n in [('quill', 7), ('ember', 9)]:
        torch.manual_seed(42 + n)
        b[x] = dict(ids=torch.randint(100, 900, (2, n)), att=torch.ones(2, n, dtype=torch.long),
                    pos=torch.arange(n)[None].expand(2, -1), tick=torch.full((2, n), -1),
                    cut=torch.full((2,), link.FAR))
        positions[x] = torch.tensor([[n - 1], [n - 2]])
    passed = []
    with torch.inference_mode():
        ref, _ = pair.joint(b)
        cut, _ = pair.joint(b, mode='zero')
        with CausalTrace(pair, positions, capture_full=True) as tr:
            got, _ = pair.joint(b)
        for x in link.SIDES:
            torch.testing.assert_close(ref[x], got[x], rtol=0, atol=0)
        passed.append('tracing bitwise preserves original fused forward')
        with CausalTrace(pair, positions, capture=False) as disabled:
            unrecorded, _ = pair.joint(b)
        for x in link.SIDES:
            torch.testing.assert_close(ref[x], unrecorded[x], rtol=0, atol=0)
        assert not disabled.states and not disabled.stats
        passed.append('disabled recording preserves forward and avoids state collection')
        assert len(tr.stats) == 12
        assert all(0 <= r['null_attention_mass'] <= 1 for r in tr.stats)
        assert all((x, l, 'native_block_out') in tr.states for x in link.SIDES for l in range(6))
        passed.append('both sides and later native layers captured at requested token positions')
        with CausalTrace(pair, positions, keep_receivers=[]) as t0:
            zero, _ = pair.joint(b)
        for x in link.SIDES:
            torch.testing.assert_close(cut[x], zero[x], rtol=0, atol=0)
        passed.append('cutting both writes equals upstream zero under same weights')
        with CausalTrace(pair, positions, keep_receivers=['quill']):
            one, _ = pair.joint(b)
        torch.testing.assert_close(cut['ember'], one['ember'], rtol=0, atol=0)
        assert not torch.equal(cut['quill'], one['quill'])
        passed.append('one-way read leaves sender identical to its own cut run')
        with CausalTrace(pair, positions, keep_blocks=[]):
            zero_layer, _ = pair.joint(b)
        for x in link.SIDES:
            torch.testing.assert_close(cut[x], zero_layer[x], rtol=0, atol=0)
        passed.append('empty retained-layer set equals cut')
        with CausalTrace(pair, positions, sender_donors=tr.full_pre, post_donors=tr.full_post):
            identity, _ = pair.joint(b)
        for x in link.SIDES:
            torch.testing.assert_close(ref[x], identity[x], rtol=0, atol=0)
        passed.append('self-donor sender and receiver patches preserve original result')
        with CausalTrace(pair, positions, sender_kv_donors=tr.full_pre, sender_gate_donors=tr.full_pre, capture=False):
            identity_parts, _ = pair.joint(b)
        for x in link.SIDES:
            torch.testing.assert_close(ref[x], identity_parts[x], rtol=0, atol=0)
        passed.append('self K/V and gate donor inputs preserve original result')
        with CausalTrace(pair, positions, write_norms=tr.full_write_norms, capture=False):
            norm_identity, _ = pair.joint(b)
        for x in link.SIDES:
            torch.testing.assert_close(ref[x], norm_identity[x], rtol=0, atol=0)
        passed.append('matching factual write norms alone preserves original result')
        with CausalTrace(pair, positions, drop_writes={('ember',l) for l in pair.xblocks}, capture=False):
            dropped, _ = pair.joint(b)
        for x in link.SIDES:
            torch.testing.assert_close(one[x], dropped[x], rtol=0, atol=0)
        passed.append('individual write removal equals the same one-way communication')
        with CausalTrace(pair, positions, scale=0):
            scaled, _ = pair.joint(b)
        for x in link.SIDES:
            torch.testing.assert_close(cut[x], scaled[x], rtol=0, atol=0)
        passed.append('zero strength equals cut')
        try:
            with CausalTrace(pair, positions):
                raise RuntimeError('intentional restoration check')
        except RuntimeError:
            pass
        restored, _ = pair.joint(b)
        for x in link.SIDES:
            torch.testing.assert_close(ref[x], restored[x], rtol=0, atol=0)
        assert '_exchange' not in pair.__dict__
        assert all('show' not in s.__dict__ and 'read' not in s.__dict__ for s in pair.bridge.values())
        passed.append('context restores original methods after exception')
    receipt = dict(evidence_type='tiny_random_infrastructure_only', status='passed', checks=passed,
                   elapsed_seconds=time.time() - start, gpu_seconds=0,
                   warning='No trained-checkpoint inference, semantic transfer, probes or mechanism claim follows from these assertions.')
    (ROOT / 'logs/CAUSAL_TRACE_CHECK.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
