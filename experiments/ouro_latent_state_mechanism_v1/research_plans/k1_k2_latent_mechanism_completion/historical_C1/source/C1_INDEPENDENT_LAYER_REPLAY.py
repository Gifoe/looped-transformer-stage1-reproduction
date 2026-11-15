"""Independent explicit-layer C1 replay, distinct from collector norm hooks.

Called with a freshly loaded, unhooked pinned Ouro model by the replay driver.
No labels, donor selection, probe fit or checkpoint write occurs here.
"""
import hashlib
import types
import torch

MODES = ('N', 'P', 'U', 'UX', 'UA', 'UXA', 'NXU', 'NAU', 'NXUAU')

def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def independent_forward(model, ids, R, prefix_len, targets, deltas, branch_refs, mode):
    assert mode in MODES
    assert not model.training and ids.ndim == 2 and ids.shape[0] == 1
    assert R == ids.shape[1]-1 and 0 < prefix_len < R
    assert len(model.model.layers) == 24
    assert set(deltas) == set(range(9, 16))
    assert set(branch_refs) == {'xN', 'aN', 'xU', 'aU'}
    assert all(v.shape == (1, 2048) and v.dtype == torch.bfloat16 for v in branch_refs.values())
    assert all(v.shape == (1, 2048) and v.dtype == torch.float64 for v in deltas.values())
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    guard = [(p, p._version) for p in model.parameters()]
    core = model.model
    old_steps = core.total_ut_steps
    saved_forward = {}
    events = {i: [] for i in (8, 9, 10, 11, 12, 13, 14, 15, 17)}
    rows = {}
    q1 = []
    on_u = mode in ('U', 'UX', 'UA', 'UXA')
    replacements = {
        'UX': ('xN', 'aU'), 'UA': ('xU', 'aN'), 'UXA': ('xN', 'aN'),
        'NXU': ('xU', 'aN'), 'NAU': ('xN', 'aU'), 'NXUAU': ('xU', 'aU')}

    def make_forward(i):
        def forward(layer, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, use_cache=False, cache_position=None,
                    position_embeddings=None, **kwargs):
            ut = kwargs.get('current_ut')
            assert type(ut) is int and ut == len(events[i]) and ut in (0, 1)
            events[i].append(ut)
            assert past_key_value is None and use_cache is False
            assert hidden_states.shape == (1, R+1, 2048)
            if i == 8:
                if ut == 0:
                    q1.append(hidden_states[:, prefix_len:R].detach().clone())
                elif mode == 'P':
                    assert len(q1) == 1
                    hidden_states = hidden_states.clone()
                    hidden_states[:, prefix_len:R] = q1[0]
            # Deliberately implement the pinned layer's computation explicitly.
            # Do not call the producer helper or norm-prehook injection helper.
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            attended, _ = layer.self_attn(
                hidden_states=normalized, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                use_cache=use_cache, cache_position=cache_position,
                position_embeddings=position_embeddings, **kwargs)
            attention_increment = layer.input_layernorm_2(attended)
            if ut == 1:
                entry = {'xR': residual[:, R].detach().clone(),
                         'actual_aR': attention_increment[:, R].detach().clone(),
                         'x_nonR_sha256': tensor_hash(residual[:, :R]),
                         'actual_a_nonR_sha256': tensor_hash(attention_increment[:, :R])}
                if on_u and 9 <= i <= 15:
                    attention_increment = attention_increment.clone()
                    delta = deltas[i].to(device=hidden_states.device)
                    attention_increment[:, R] = (attention_increment[:, R].double()-delta).to(torch.bfloat16)
                entry['used_aR'] = attention_increment[:, R].detach().clone()
            hidden_states = residual + attention_increment
            if i == 17 and ut == 1:
                entry['actual_yR'] = hidden_states[:, R].detach().clone()
                entry['actual_y_nonR_sha256'] = tensor_hash(hidden_states[:, :R])
                if mode in replacements:
                    xkey, akey = replacements[mode]
                    x = branch_refs[xkey].to(device=hidden_states.device)
                    a = branch_refs[akey].to(device=hidden_states.device)
                    hidden_states = hidden_states.clone()
                    hidden_states[:, R] = x + a
                    entry['selected_xR'] = x.detach().clone()
                    entry['selected_aR'] = a.detach().clone()
                else:
                    entry['selected_xR'] = residual[:, R].detach().clone()
                    entry['selected_aR'] = attention_increment[:, R].detach().clone()
                entry['selected_yR'] = hidden_states[:, R].detach().clone()
                assert torch.equal(entry['selected_yR'], entry['selected_xR']+entry['selected_aR'])
                assert tensor_hash(hidden_states[:, :R]) == entry['actual_y_nonR_sha256']
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = layer.post_attention_layernorm_2(hidden_states)
            if i == 17 and ut == 1:
                entry['actual_skipR'] = residual[:, R].detach().clone()
                assert torch.equal(entry['actual_skipR'], entry['selected_yR'])
                entry['mlp_incrementR'] = hidden_states[:, R].detach().clone()
            hidden_states = residual + hidden_states
            if ut == 1:
                entry['post_mlpR'] = hidden_states[:, R].detach().clone()
                entry['post_mlp_nonR_sha256'] = tensor_hash(hidden_states[:, :R])
                rows[i] = entry
            return hidden_states
        return forward

    try:
        for i in events:
            layer = core.layers[i]
            assert not layer._forward_hooks and not layer._forward_pre_hooks
            for mod in layer.modules():
                assert not mod._forward_hooks and not mod._forward_pre_hooks
            saved_forward[i] = layer.forward
            layer.forward = types.MethodType(make_forward(i), layer)
        core.total_ut_steps = 2
        with torch.inference_mode():
            output = model(ids, labels=None, use_cache=False, exit_at_step=1)
            logits = output.logits[:, R].detach().clone()
        assert all(v == [0, 1] for v in events.values())
        assert len(q1) == 1 and set(rows) == set(events)
        assert logits.shape == (1, 49152) and logits.dtype == torch.bfloat16
        assert torch.isfinite(logits).all()
        candidate = [float(logits[0, t]) for t in targets]
        scores = [float(logits.float().log_softmax(-1)[0, t]) for t in targets]
        return {'mode':mode, 'logits':logits.cpu(), 'full_vocab_R_sha256':tensor_hash(logits),
                'candidate_logits':candidate, 'scores':scores, 'events':events,
                'tensors':{i:{k:(v.cpu() if isinstance(v,torch.Tensor) else v) for k,v in row.items()} for i,row in rows.items()},
                'implementation':'Explicit pinned-layer computation and two-branch addition; no collector injection helper.'}
    finally:
        for i, original in saved_forward.items():
            core.layers[i].forward = original
        core.total_ut_steps = old_steps
        assert all(p._version == version for p, version in guard)
