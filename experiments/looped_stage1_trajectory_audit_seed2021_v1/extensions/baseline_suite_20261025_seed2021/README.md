# Training-free / Looped Transformer baseline extension

This is an extension of the sealed Stage1 audit. Its active experiment protocol
is `configs/protocol.json`; historical Stage1 artifacts are read-only.

Models: Qwen3-4B-Base, Ouro-1.4B and Ouro-2.6B. Added comparisons include source
audited DoLa on Qwen, fixed/adaptive LoopCD, projection/temperature controls,
validation-selected halting, actual early-exit latency and autoregressive math.

Read `FINDINGS.md` after completion. Until then, `STATUS.json`, the active process
handle and `logs/*supervisor*` identify progress. A timed-out observation is not
proof that a worker stopped. Do not launch a second GPU owner.

The official Training-Free Looped Transformer repository currently returns 404.
Qwen loops are an independent recipe with model/cache parity checks, not a claim
of direct author-patch parity. DoLa-on-Qwen and Ouro-Hidden are explicitly scoped
adaptations. Full paper headline results are outside this benchmark subset.

The extension has its own cumulative GPU ledger including Stage1. The hard
50-hour cap includes failures, pilots and model loading. No model training or
quantization is performed. See `REPRODUCE.md` for source pins and commands.
