# Experiment log

1. Pinned the official zero-shot ARC-C template and scored 299 validation questions, candidates in official order. Tested compact projection against the model wrapper at each depth. One sampled batch agreed on all per-token argmaxes and had a maximum 0.125 logit difference across BF16 GEMM shapes.
2. Compared K4 Q/K/V teacher replacements with native and dose-matched controls in a predeclared four-layer window. The first hook setup selected a different mask path and failed parity. The delivered hook keeps the native SDPA configuration and passes bitwise identity.
3. Screened clipped secant vs the validation-selected depth-matched references. Development acc_norm changed by -1.67 and +0.00 pp at K3 and K4. Neither passed the gate. Froze source and scores before scoring test.
4. Executed the complete 1,172-item test for all frozen variants. Followed up the oracle intervention on all 171 unused validation IDs with the same interventions and both controls. No oracle passed both development stages, and no training was triggered.
5. Measured actual cached prefill and one-token decode. The checkpoint's default UniversalTransformerCache and a non-sized fallback failed against the installed Transformers. The recorded run explicitly sizes DynamicCache and checks the cache advances and the four native top token predictions match full recomputation.
6. Measured standalone full-test wall time at each native depth and for the selected low-cost comparison. Archived scores, timings, resampling statistics, configuration hashes and non-promoted methods.
