# Ouro cache scorer audit

**Compatibility update after GB10 readiness:** the actual HF 4.55.4 `Cache.key_cache`
property is read-only, so the pinned `UniversalTransformerCache` constructor fails.
The scorer now explicitly creates standard HF `DynamicCache`, which accepts virtual
loop-times-layer indices, and forks both the cache object and every `DynamicLayer`
object while sharing immutable KV tensors. It adds a `layer_class_to_replicate`
compatibility marker only when absent, preventing the pinned model from replacing
this usable cache with its broken custom constructor. `_ouro_capacity` records the
required recurrent slot count. Parent checks use actual layer `keys`/`values`.
This supersedes the custom-cache creation/fork description below for real runs.
The change is based on server-extracted API source, with no modification to pinned
model source or weights. Real GB10 numerical parity must be rerun after this fix.

Implemented `scripts/ouro_cached_scoring.py`, a torch-only, path-free scoring module. It imports no old pipeline code, reads no files, and launches no jobs. Every forward/scoring public method is under `torch.no_grad`; all modules must be in eval mode. Default precision guard requires BF16 model parameters and BF16 LM-head outputs. Vocabulary log-softmax and summation use FP32. Character normalization remains the caller's original lm-eval rule.

## Integration

```python
scorer = OuroCachedScorer(model, kmax=6)
# Strictly identical token IDs from original _encode_pair; fewer than every query.
shared = scorer.build_shared_prefix(demo_prefix_token_ids)
pairs = [(encoded_prefix_1, continuation_1), (encoded_prefix_2, continuation_2)]
parity = scorer.compare_cached_uncached(pairs, shared, char_lengths=choice_char_lengths)
depth_check = scorer.check_depth_invariance(pairs)
scores = scorer.score_candidates(pairs, shared=shared,
    capture_raw=True, return_states=True)
```

Run parity on synthetic/non-test examples before substantive use, including multi-token answers, different encoded prefixes, and long 25-shot contexts. Report actual token/sum/character-normalized discrepancies and prediction matches; the default 0.05 screening tolerance is not proof that cached and uncached BF16 computations are interchangeable. Freeze a justified tolerance and reject material prediction or margin changes. Every new intervention also needs cached-vs-uncached validation.

`sum_loglikelihood` uses integer keys 1..6. `loopcd` gives fixed omega=0.5 and adaptive cap=1 at K4 vs K1 with full-vocabulary softmax. `token_loglikelihood` permits precise error audits. Optional raw and normalized CPU tensors have shape `[1, sampled_continuation_positions, hidden_size]`, one tensor per loop; `state_target_indices` records positions. No final normalization is applied again to normalized states. `capture_raw=True` with `return_states=False` computes raw captures but discards them; ordinarily enable both together.

Shared demo caches retain no hidden-state dump. A query cache retains only its last input position (prediction of continuation token zero), and target-prefix processing retains only positions predicting continuation tokens 1..N-1. All candidate target tokens contribute to scores. Query caches are grouped by the full actual encoded prefix tuple, and released after `score_candidates`. If a query equals the entire shared cache prefix, the scorer rebuilds that query to obtain its missing final prediction state; correctness takes priority over reuse. No truncation or padding is performed.

## Cache and intervention constraints

The pinned `UniversalTransformerCache.update` replaces list entries using `torch.cat`; it does not modify cached tensors in place. Forking creates new `key_cache`, `value_cache` and `layers` lists, preserving tensor references, `_seen_tokens`, and `max_cache_size`. The scorer supplies absolute `cache_position` and `position_ids`, and a total-length attention mask. It checks all Kmax times layer-count cache slots have the expected prefix length and checks parent list/tensor identities, versions and shapes after child forwards.

Scorer construction sets `model.model.total_ut_steps`, `model.config.total_ut_steps`, and `model.model.config.total_ut_steps` consistently. Separate-depth checks restore all fields even on error.

**After any edit or activation intervention change:** call `scorer.invalidate()` AFTER the change, discard all parent handles, and rebuild the shared prefix under the new computation. Parameter identities/versions/storage and registered hook identities are checked as a supplementary safeguard. Old handles carry an epoch and are rejected after invalidation. `.data` mutation and changes inside a hook closure cannot be detected reliably; explicit invalidation is mandatory. Baseline caches must never be reused under an edit, even if the edited physical layer runs mainly at a later loop.

Hooks must be causal and invariant to chunking. A hook that modifies “the last token of this forward,” or uses a full-sequence statistic, can make cached and uncached executions different mathematical interventions. Define positions globally via the forward's explicit positions or use truly position-independent weight-equivalent hooks. Dynamic length-dependent positional schemes also require separate scrutiny; this implementation is intended for the pinned checkpoint, not arbitrary Ouro variants.

## Checks actually executed

- Python compilation passed.
- CPU BF16 synthetic causal recurrent model, using the actual pinned `UniversalTransformerCache` class: three candidates including shared/differing query prefixes and one-/multi-token continuations; cached versus whole-sequence score/token differences were zero; separate K3/4/5 versus K6 intermediates were exact.
- Synthetic checks passed for parent-cache preservation, raw/normalized continuation-state shapes, a query equal to its shared prefix, stale-epoch rejection, changed-hook rejection, changed-weight rejection, and successful reconstruction after invalidation.
- A tiny native Ouro CPU forward could not be completed with the local installed Transformers version: config compatibility failed first on `pad_token_id`, then on missing `ROPE_INIT_FUNCTIONS['default']`. No pinned sources or dependencies were modified to bypass this. Synthetic success is a scorer/control-flow check, NOT native checkpoint parity.

No GPU/remote jobs or benchmark-data reads were performed. Native GB10 checkpoint parity, real attention backend behavior, BF16 chunk-size differences, and practical throughput remain the main agent's required checks before enabling cache reuse for the experiment.
