"""Explicit decoder-layer replay, independent of AB1 injection hooks.

Preparation only: pinned source hash, saved artifact integrity, real GPU acceptance,
independent control generation and runtime/SDPA audit are caller responsibilities.
"""
import hashlib
import sys
import torch

if not __debug__:
    raise RuntimeError('Python optimization disables acceptance checks')

EDGES = {'Q_R_E': ('H00', 'H10', 'Q'), 'Q_R_B': ('H11', 'H01', 'Q'),
         'R_Q_E': ('H00', 'H01', 'R'), 'R_Q_B': ('H11', 'H10', 'R')}

def bitwise(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def _move(x, reference):
    if x.dtype != reference.dtype or x.shape != reference.shape or not torch.isfinite(x).all():
        raise ValueError('invalid archived tensor')
    return x.to(reference.device)

def make_boundary(E, B, prefix, R, name, stored):
    if E.shape != B.shape or E.ndim != 3 or E.shape[0] != 1 or not 0 < prefix < R == E.shape[1]-1:
        raise ValueError('invalid boundary or regions')
    H = {}
    for tag, q, r in [('H00', E, E), ('H10', B, E), ('H01', E, B), ('H11', B, B)]:
        x = B.clone(); x[:, prefix:R] = q[:, prefix:R]; x[:, R] = r[:, R]; H[tag] = x
    if name in H:
        return H[name]
    for edge, (receiver, donor, region) in EDGES.items():
        if name in (edge+'_norm', edge+'_random'):
            x = H[receiver].clone(); sl = slice(prefix, R) if region == 'Q' else slice(R, R+1)
            x[:, sl] = _move(stored[name+'_injected_region'], x[:, sl])
            outside = torch.ones(R+1, dtype=torch.bool, device=x.device); outside[sl] = False
            if not bitwise(x[:, outside], H[receiver][:, outside]): raise AssertionError('control leaked outside region')
            return x
    if name in ('U_Q9_N', 'N_Q9_U'):
        return H['H01' if name == 'U_Q9_N' else 'H11']
    if name == 'E_all': return E.clone()
    raise KeyError(name)

@torch.inference_mode()
def replay(model, ids, prefix, R, name, stored, *, mask_factory=None):
    """Run both recurrent loops by calling physical layers, without injection hooks.

    mask_factory, if supplied, has same kwargs as the pinned module's
    create_causal_mask and must return its actual full-attention mask. This is an
    injection-independent audit, not an independently implemented decoder.
    """
    body = model.model
    if model.training or body.total_ut_steps != 2 or len(body.layers) != 24 or body.config.num_hidden_layers != 24:
        raise ValueError('requires evaluation, two loops, 24 layers')
    if ids.ndim != 2 or ids.shape != (1, R+1) or not 0 < prefix < R:
        raise ValueError('invalid input regions')
    if getattr(body, 'has_sliding_layers', False): raise ValueError('unsupported sliding layers')
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise RuntimeError('replay must run without module hooks')
    x = body.embed_tokens(ids); E = x.clone(); cp = torch.arange(ids.shape[1], device=ids.device); pos = cp.unsqueeze(0)
    source_module = sys.modules[type(body).__module__]
    create_mask = mask_factory or source_module.create_causal_mask
    mask = create_mask(config=body.config, input_embeds=x, attention_mask=None, cache_position=cp, past_key_values=None, position_ids=pos)
    rope = body.rotary_emb(x, pos)
    state = []; gates = []; steps = []; q_before = None
    for ut in range(2):
        if ut == 1:
            B = x.clone()
            if not bitwise(E, _move(stored['source_E'], E)) or not bitwise(B, _move(stored['source_B'], B)):
                raise AssertionError('fresh boundary differs from archived source')
            x = make_boundary(E, B, prefix, R, name, stored)
        for index, layer in enumerate(body.layers):
            if getattr(layer, 'attention_type', None) != 'full_attention': raise ValueError('unexpected attention type')
            if ut == 1 and index == 8:
                q_before = x.clone()
                if name in ('U_Q9_N', 'N_Q9_U'):
                    receiver_source = stored['U_Q9_full' if name == 'U_Q9_N' else 'native_Q9_full']
                    if not bitwise(x, _move(receiver_source, x)): raise AssertionError('Q9 receiver differs from archived full state')
                    source = stored['native_Q9_full' if name == 'U_Q9_N' else 'U_Q9_full']
                    donor = _move(source, x)
                    baseline = x.clone(); x = x.clone(); x[:, prefix:R] = donor[:, prefix:R]
                    if not bitwise(x[:, :prefix], baseline[:, :prefix]) or not bitwise(x[:, R:], baseline[:, R:]):
                        raise AssertionError('Q9 altered receiver P/R')
            x = layer(x, attention_mask=mask, position_ids=pos, past_key_value=None,
                      use_cache=False, cache_position=cp, position_embeddings=rope, current_ut=ut)
            if not torch.is_tensor(x) or not torch.isfinite(x).all(): raise RuntimeError('invalid layer output')
            steps.append((ut, index))
        x = body.norm(x); state.append(x.clone()); gates.append(body.early_exit_gate(x).clone())
    if steps != [(u, i) for u in range(2) for i in range(24)]: raise AssertionError('layer coverage')
    z1 = model.lm_head(state[0])[:, R, :]; z2 = model.lm_head(state[1])[:, R, :]
    if z1.ndim != 2 or z2.shape != z1.shape or not torch.isfinite(z2).all(): raise AssertionError('head shape')
    expected = stored['native_K1_logits'] if name == 'E_all' else stored[name+'_logits']
    if not bitwise(z1, _move(stored['native_K1_logits'], z1)) or not bitwise(z2, _move(expected, z2)):
        raise AssertionError('full logits differ from collector')
    if name == 'H11' and not bitwise(z2, _move(stored['native_K2_logits'], z2)):
        raise AssertionError('H11 differs from native K2')
    return {'condition': name, 'layers': steps, 'logits_sha256': digest(z2),
            'gate_sha256': [digest(g) for g in gates], 'Q9_receiver_sha256': digest(q_before),
            'mask_sha256': None if mask is None else digest(mask),
            'position_sha256': digest(pos), 'cache_position_sha256': digest(cp),
            'rope_sha256': [digest(t) for t in rope], 'use_cache': False,
            'scope': 'explicit layer replay; same decoder and mask factory; no injection hooks'}

