"""Dataset/path-free, batch-one teacher-forced scoring for pinned Ouro.

Caller supplies lm-eval _encode_pair results, one (prefix_ids, target_ids) per
candidate. Shared-prefix tokens MUST be identical token IDs, not text prefixes.
Call invalidate() after EVERY weight/intervention change, then rebuild caches.
Hooks must be causal, position-aware and chunk-invariant for cache equivalence.
No runtime parity claim is implied: use compare_cached_uncached on non-test data.
"""
from dataclasses import dataclass
import copy
from typing import Any, Optional

import torch


def _version(t):
    try:
        return t._version
    except RuntimeError:
        return None


def _tensor_mark(t):
    return None if t is None else (id(t), t.data_ptr(), tuple(t.shape), _version(t))


def _cache_entries(cache):
    if getattr(cache, '_ouro_dynamic', False):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def _cache_mark(cache):
    return (tuple((_tensor_mark(k), _tensor_mark(v)) for k, v in _cache_entries(cache)),
            cache.get_seq_length(), getattr(cache, '_ouro_capacity', getattr(cache, 'max_cache_size', None)))


def _new_dynamic_cache(capacity):
    # HF 4.55.4 Cache.key_cache is read-only, breaking the pinned custom cache's
    # constructor. Standard DynamicCache handles loop*layer indices without a cap.
    # This marker also satisfies pinned needs_universal_cache's dynamic-cache test
    # on API versions exposing layer_classes instead of layer_class_to_replicate.
    from transformers.cache_utils import DynamicCache, DynamicLayer
    cache = DynamicCache()
    if getattr(cache, 'layer_class_to_replicate', None) is None:
        cache.layer_class_to_replicate = DynamicLayer
    cache._ouro_dynamic = True
    cache._ouro_capacity = capacity
    return cache


def fork_universal_cache(cache):
    """Fork dynamic layer OBJECTS as well as their list; share immutable KV tensors.

    Keeps a legacy custom-cache branch for standalone synthetic checks. Real HF
    4.55.4 forwards use standard DynamicCache, never the broken custom constructor.
    """
    if getattr(cache, '_ouro_dynamic', False):
        fork = copy.copy(cache)
        fork.layers = [copy.copy(layer) for layer in cache.layers]
        return fork
    required = ('key_cache', 'value_cache', '_seen_tokens', 'max_cache_size', 'layers')
    if any(not hasattr(cache, name) for name in required):
        raise TypeError('Expected pinned Ouro UniversalTransformerCache')
    fork = type(cache)(max_cache_size=cache.max_cache_size)
    fork.key_cache = list(cache.key_cache)
    fork.value_cache = list(cache.value_cache)
    fork.layers = list(cache.layers)
    fork._seen_tokens = cache._seen_tokens
    return fork


@dataclass(frozen=True)
class PrefixCache:
    owner: object
    epoch: int
    tokens: tuple
    cache: Any
    normalized: Optional[tuple] = None  # one last-position tensor per loop
    raw: Optional[tuple] = None


class OuroCachedScorer:
    """Single model, batch one, no padding; not thread-safe.

    Sets model depth to kmax; caller owns checkpoint/loading/seed/attention backend.
    Parent caches are explicit handles: no unbounded internal query-cache store.
    invalidate() makes ALL handles stale, including handles held by the caller.
    Parameter-version checks are supplementary; .data writes or closure changes
    cannot be reliably detected. Those require explicit invalidate().
    """
    def __init__(self, model, kmax=6, require_bf16=True):
        if kmax < 1:
            raise ValueError('kmax must be positive')
        self.model, self.kmax = model, int(kmax)
        self.owner, self.epoch = object(), 0
        self.require_bf16 = require_bf16
        model.eval()
        self._set_depth(self.kmax)
        self._stamp = self._fingerprint()
        self._check_model()

    def _set_depth(self, depth):
        self.model.model.total_ut_steps = depth
        self.model.config.total_ut_steps = depth
        self.model.model.config.total_ut_steps = depth

    def _fingerprint(self):
        params = tuple((n, _tensor_mark(p), str(p.dtype), str(p.device))
                       for n, p in self.model.named_parameters())
        hooks = tuple((n, tuple((k, id(v)) for k, v in m._forward_hooks.items()),
                       tuple((k, id(v)) for k, v in m._forward_pre_hooks.items()))
                      for n, m in self.model.named_modules()
                      if m._forward_hooks or m._forward_pre_hooks)
        return params, hooks

    def _check_model(self):
        if any(m.training for m in self.model.modules()):
            raise RuntimeError('Scoring requires eval() on all modules')
        if any(v != self.kmax for v in (self.model.model.total_ut_steps,
               self.model.config.total_ut_steps, self.model.model.config.total_ut_steps)):
            raise RuntimeError('Model/config loop depth changed; reset and invalidate')
        if self.require_bf16 and any(p.is_floating_point() and p.dtype != torch.bfloat16
                                    for p in self.model.parameters()):
            raise RuntimeError('Expected BF16 parameters throughout model')
        if self._fingerprint() != self._stamp:
            raise RuntimeError('Weights or hooks changed: call invalidate(), rebuild prefixes')

    def invalidate(self):
        """Call AFTER changing weights/hooks/closure coefficients; rebuild every prefix."""
        self.epoch += 1
        self.model.eval()
        self._set_depth(self.kmax)
        self._stamp = self._fingerprint()

    @property
    def device(self):
        return self.model.model.embed_tokens.weight.device

    def _check_prefix(self, prefix):
        if prefix.owner is not self.owner or prefix.epoch != self.epoch:
            raise RuntimeError('Foreign or invalidated prefix cache; rebuild under current model')
        expected = self.kmax * self.model.model.config.num_hidden_layers
        capacity = getattr(prefix.cache, '_ouro_capacity', getattr(prefix.cache, 'max_cache_size', None))
        if capacity != expected:
            raise RuntimeError('Wrong recurrent cache capacity')
        entries = _cache_entries(prefix.cache)
        if len(entries) != expected:
            raise RuntimeError('Incomplete recurrent cache')
        for k, v in entries:
            if k is None or v is None or k.shape[-2] != len(prefix.tokens) or v.shape[-2] != len(prefix.tokens):
                raise RuntimeError('Prefix cache length mismatch')

    @torch.no_grad()
    def _run(self, tokens, parent=None, positions=(), capture_raw=False, use_cache=True):
        """positions are LOCAL input positions retained; all others are discarded."""
        if not tokens:
            raise ValueError('Cannot forward zero tokens')
        self._check_model()
        start, cache, parent_mark = 0, None, None
        if use_cache:
            cache = _new_dynamic_cache(self.kmax * self.model.model.config.num_hidden_layers)
        if parent is not None:
            self._check_prefix(parent)
            start = len(parent.tokens)
            parent_mark = _cache_mark(parent.cache)
            cache = fork_universal_cache(parent.cache)
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        selected = torch.tensor(list(positions), dtype=torch.long, device=self.device)
        raw = []
        handle = None
        if capture_raw and selected.numel():
            def collect(module, args):
                raw.append(args[0].index_select(1, selected).detach().clone())
            handle = self.model.model.norm.register_forward_pre_hook(collect)
        try:
            if not use_cache:
                if parent is not None:
                    raise ValueError('Uncached original call cannot have a parent cache')
                # EXACT legacy score_arc.py call. Supplying an all-ones mask
                # changes HF4.55 SDPA mask/GQA paths even with no padding.
                outputs, states, _ = self.model.model(input_ids=x, use_cache=False)
            else:
                outputs, states, _ = self.model.model(
                    input_ids=x, use_cache=True, past_key_values=cache,
                    attention_mask=torch.ones((1, start + len(tokens)), dtype=torch.long, device=self.device),
                    cache_position=torch.arange(start, start + len(tokens), device=self.device),
                    position_ids=torch.arange(start, start + len(tokens), device=self.device).unsqueeze(0))
            if len(states) != self.kmax:
                raise RuntimeError('Unexpected recurrent state count')
            normalized = tuple(s.index_select(1, selected).detach().clone() for s in states) if selected.numel() else ()
            if capture_raw and selected.numel() and len(raw) != self.kmax:
                raise RuntimeError('Boundary norm hook count differs from loop count')
            if parent is not None and _cache_mark(parent.cache) != parent_mark:
                raise RuntimeError('Reusable parent cache was mutated')
            return outputs.past_key_values, normalized, tuple(raw)
        finally:
            if handle is not None:
                handle.remove()

    @torch.no_grad()
    def build_shared_prefix(self, token_ids):
        """Build shared demo KV only: retains NO prompt hidden states. Empty -> None.

        Use fewer tokens than EVERY candidate prefix; final query token must be
        processed when creating its prediction state. score_candidates handles an
        accidentally equal shared prefix safely by rebuilding that query uncached.
        """
        tokens = tuple(int(t) for t in token_ids)
        if not tokens:
            return None
        cache, _, _ = self._run(tokens)
        return PrefixCache(self.owner, self.epoch, tokens, cache)

    @torch.no_grad()
    def build_query_prefix(self, prefix_ids, shared=None, capture_raw=False):
        tokens = tuple(int(t) for t in prefix_ids)
        if not tokens:
            raise ValueError('Each encoded prefix must be nonempty')
        if shared is not None:
            self._check_prefix(shared)
            if tokens[:len(shared.tokens)] != shared.tokens:
                raise ValueError('Shared token IDs are not a prefix of this candidate prefix')
            if len(tokens) == len(shared.tokens):
                shared = None  # no stored final state in demo KV; correctness first
        offset = 0 if shared is None else len(shared.tokens)
        tail = tokens[offset:]
        cache, normalized, raw = self._run(tail, shared, [len(tail)-1], capture_raw)
        return PrefixCache(self.owner, self.epoch, tokens, cache, normalized,
                           raw if capture_raw else None)

    @torch.no_grad()
    def _score_states(self, states, targets, raw=(), loopcd=True, return_states=False,
                      sample_positions=None):
        targets = torch.tensor(list(targets), dtype=torch.long, device=self.device)
        logits = []
        sums = {}
        token_scores = {}
        for k, state in enumerate(states, 1):
            z = self.model.lm_head(state)[0]  # preserve native BF16 head output
            if self.require_bf16 and z.dtype != torch.bfloat16:
                raise RuntimeError('LM head output is not BF16')
            z = z.float()
            lp = z.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
            if not torch.isfinite(lp).all():
                raise RuntimeError('Nonfinite continuation log likelihood')
            sums[k], token_scores[k] = float(lp.sum()), lp.cpu().tolist()
            if loopcd and k in (1, 4):
                logits.append(z)
        result = dict(sum_loglikelihood=sums, token_loglikelihood=token_scores,
                      target_token_count=len(targets), loopcd={})
        if loopcd and self.kmax >= 4:
            early, final = logits
            fixed = final + 0.5 * (final - early)
            top = final.softmax(-1).topk(2, dim=-1).values
            strength = 1 - (top[:, 0] - top[:, 1])
            adaptive = final + strength[:, None] * (final - early)
            for name, z in [('fixed', fixed), ('adaptive', adaptive)]:
                lp = z.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
                result['loopcd'][name] = float(lp.sum())
            result['adaptive_strength'] = strength.cpu().tolist()
        if return_states:
            idx = list(range(len(targets))) if sample_positions is None else list(sample_positions)
            if any(i < 0 or i >= len(targets) for i in idx):
                raise ValueError('State sample positions must index continuation predictions')
            sel = torch.tensor(idx, device=self.device, dtype=torch.long)
            result['state_target_indices'] = idx
            result['normalized_states'] = tuple(s.index_select(1, sel).detach().cpu() for s in states)
            result['raw_states'] = tuple(s.index_select(1, sel).detach().cpu() for s in raw)
        return result

    @torch.no_grad()
    def score_candidates(self, encoded_pairs, shared=None, capture_raw=False,
                         return_states=False, sample_positions=None, loopcd=True):
        """Returns candidate-order list; NO length normalization or argmax here.

        encoded_pairs: [(prefix_ids, continuation_ids), ...]; prefixes can differ.
        sample_positions indexes prediction positions, not input token positions.
        Query caches are reused within this call and released on return.
        """
        self._check_model()
        queries, results = {}, []
        for prefix_ids, target_ids in encoded_pairs:
            prefix = tuple(int(t) for t in prefix_ids)
            target = tuple(int(t) for t in target_ids)
            if not target:
                raise ValueError('Each continuation must contain at least one token')
            if prefix not in queries:
                queries[prefix] = self.build_query_prefix(prefix, shared, capture_raw)
            query = queries[prefix]
            states, raw = query.normalized, query.raw or ()
            if len(target) > 1:
                _, tail, raw_tail = self._run(target[:-1], query, range(len(target)-1), capture_raw)
                states = tuple(torch.cat([a, b], dim=1) for a, b in zip(states, tail))
                if capture_raw:
                    raw = tuple(torch.cat([a, b], dim=1) for a, b in zip(raw, raw_tail))
            results.append(self._score_states(states, target, raw, loopcd, return_states, sample_positions))
        return results

    @torch.no_grad()
    def score_uncached(self, encoded_pairs, capture_raw=False, return_states=False,
                       sample_positions=None, loopcd=True):
        """Native whole-sequence oracle; no prefix cache and no previous results."""
        results = []
        for prefix_ids, target_ids in encoded_pairs:
            prefix, target = tuple(prefix_ids), tuple(target_ids)
            if not prefix or not target:
                raise ValueError('Nonempty prefix and continuation required')
            positions = range(len(prefix)-1, len(prefix)+len(target)-1)
            _, states, raw = self._run(prefix+target[:-1], positions=positions,
                                      capture_raw=capture_raw, use_cache=False)
            results.append(self._score_states(states, target, raw, loopcd, return_states, sample_positions))
        return results

    @torch.no_grad()
    def compare_cached_uncached(self, encoded_pairs, shared=None, char_lengths=None,
                                score_atol=0.05, token_atol=0.05):
        """NON-TEST parity diagnostic; tolerances are explicit, not auto-adjusted.

        Returns actual discrepancies and prediction equality. Defaults are coarse
        harness-compatible screening tolerances, not a scientific equivalence claim.
        Caller must examine margin flips and freeze tighter justified tolerances.
        """
        pairs = list(encoded_pairs)
        if not pairs:
            raise ValueError('At least one candidate required for parity')
        cached = self.score_candidates(pairs, shared=shared)
        full = self.score_uncached(pairs)
        lengths = [1] * len(pairs) if char_lengths is None else list(char_lengths)
        if len(lengths) != len(pairs) or any(x <= 0 for x in lengths):
            raise ValueError('One positive character length per candidate required')
        detail, predictions = [], {}
        for k in range(1, self.kmax+1):
            a = [r['sum_loglikelihood'][k] / n for r, n in zip(cached, lengths)]
            b = [r['sum_loglikelihood'][k] / n for r, n in zip(full, lengths)]
            predictions[k] = dict(cached=max(range(len(a)), key=a.__getitem__),
                                  uncached=max(range(len(b)), key=b.__getitem__))
            for c, (rc, rf) in enumerate(zip(cached, full)):
                tc, tf = rc['token_loglikelihood'][k], rf['token_loglikelihood'][k]
                detail.append(dict(k=k, candidate=c, score_abs_error=abs(a[c]-b[c]),
                                   sum_abs_error=abs(rc['sum_loglikelihood'][k]-rf['sum_loglikelihood'][k]),
                                   max_token_abs_error=max(abs(x-y) for x, y in zip(tc, tf))))
        loopcd = {name: max(abs(a['loopcd'][name]-b['loopcd'][name]) / n
                           for a, b, n in zip(cached, full, lengths))
                  for name in ('fixed', 'adaptive')} if self.kmax >= 4 else {}
        loopcd_predictions = {}
        for name in loopcd:
            a = [r['loopcd'][name] / n for r, n in zip(cached, lengths)]
            b = [r['loopcd'][name] / n for r, n in zip(full, lengths)]
            loopcd_predictions[name] = dict(cached=max(range(len(a)), key=a.__getitem__),
                                           uncached=max(range(len(b)), key=b.__getitem__))
        return dict(passed=all(d['score_abs_error'] <= score_atol and d['max_token_abs_error'] <= token_atol for d in detail)
                    and all(v['cached'] == v['uncached'] for v in predictions.values())
                    and all(v['cached'] == v['uncached'] for v in loopcd_predictions.values())
                    and all(v <= score_atol for v in loopcd.values()),
                    score_atol=score_atol, token_atol=token_atol,
                    normalized_by='character_length' if char_lengths is not None else 'none',
                    detail=detail, predictions=predictions, loopcd_predictions=loopcd_predictions,
                    loopcd_max_score_abs_error=loopcd)

    @torch.no_grad()
    def check_depth_invariance(self, encoded_pairs, depths=(3, 4, 5), atol=0.0):
        """Compare whole-sequence separate-K scores to intermediate Kmax scores.

        Does not touch any prefix caches; all depth/config fields restored on error.
        Tolerance zero deliberately reveals even tiny implementation differences.
        """
        pairs = list(encoded_pairs)
        self._check_model()
        original_k = self.kmax
        reference = self.score_uncached(pairs, loopcd=False)
        detail = []
        try:
            for depth in depths:
                if not 1 <= depth <= original_k:
                    raise ValueError('Requested depth outside Kmax')
                self.kmax = depth
                self._set_depth(depth)
                actual = self.score_uncached(pairs, loopcd=False)
                for i, (a, b) in enumerate(zip(actual, reference)):
                    detail.append(dict(k=depth, candidate=i,
                        sum_abs_error=abs(a['sum_loglikelihood'][depth]-b['sum_loglikelihood'][depth]),
                        max_token_abs_error=max(abs(x-y) for x, y in zip(a['token_loglikelihood'][depth], b['token_loglikelihood'][depth]))))
        finally:
            self.kmax = original_k
            self._set_depth(original_k)
        return dict(passed=all(d['sum_abs_error'] <= atol and d['max_token_abs_error'] <= atol for d in detail),
                    atol=atol, detail=detail)
