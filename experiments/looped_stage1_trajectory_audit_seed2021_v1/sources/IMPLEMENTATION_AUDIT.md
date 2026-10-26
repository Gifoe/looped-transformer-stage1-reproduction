# Source and implementation audit

Checked on 2026-10-23 (Asia/Shanghai).

## Qwen block wrapper

Author repository and raw file: repeated HTTP 404 through GitHub API, connector,
and web access. A cached author README is readable at
https://github.com/L-z-Chen/Training-Free-Looped-Transformer/blob/main/README.md .
The README specifies natural=g(x0), alpha=1/K, K damped Euler updates, followed by
beta*natural+(1-beta)*xK, `cache_strategy=first`, and decode bypass for likelihood.

Cross-check source: independent `JegZheng/tflt-dev`, commit
`a680f7d8d6d4f50029ece4f45b38e2c50effb9bd`.
Downloaded files under this directory are reference material and are not imported
by the evaluator. Their cache and formula code was inspected. They are not author
code. No direct equivalence to the unavailable author patch is claimed.

Our independent implementation is `scripts/qwen_recipe.py`. It keeps HF's forward,
attention masks and RoPE and redirects only layers 15-18. No-cache body calls are
followed, when caching is enabled, by one canonical write pass from the pre-loop
input. Decode bypass executes the ordinary window once. Formal likelihood calls
set use_cache=False, so the stash is inactive; explicit cache tests exercise it.

Configuration matches A0-A4 numerically. Required gates: original/HF parity,
K=1 for three strategies, beta=1, and cache length/identity tests for prefill and
incremental decode. Results remain an independent recipe reproduction even when
all these tests pass. Direct author-patch parity remains unavailable.

## Ouro

Checkpoint revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Official custom model source is preserved as `modeling_ouro.py`; the executable
copy is the pinned model snapshot. 24 shared layers, 2048 hidden dimensions,
49152 vocabulary entries, four iterations. The output norm is applied after each
iteration and its result is fed into the next iteration. Returned depth states
go directly into lm_head. A pre-norm hook is used for hidden guidance, followed
by exactly one output norm and lm_head.

`exit_at_step` selects a 0-based readout but still computes all iterations.
Changing model.total_ut_steps physically reduces depth; this is used for
separate timing tests and complete fixed-depth mathematical generations.
Both the model attribute and configuration are changed, and cache depth is
checked at 24/48/72 levels. Offline halting does not establish acceleration.
