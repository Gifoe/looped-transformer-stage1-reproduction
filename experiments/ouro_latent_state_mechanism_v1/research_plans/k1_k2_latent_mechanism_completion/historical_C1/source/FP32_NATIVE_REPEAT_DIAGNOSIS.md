# FP32 native-repeat technical diagnosis

2026-11-14, executed by current Sol under root's explicit native-repeat authorization. Original failed `C1_FP32_NUMERICAL.py` (SHA `93391406c417a51290bf60434eb64bbb3645e6f62130f333e3bde268aef22f87`), original freeze, failure log and BF16 producer were not changed. No gold was read, no scientific intervention was evaluated, and no full-eight sensitivity run was launched by this diagnostic.

Diagnostic source `C1_FP32_NATIVE_REPEAT_DIAG.py` SHA `1871b6b4cba0f3bf00f418b6a3f67ff2f3f8307ffdcff10203fb83377d9c6314` selects only the already frozen first ID `Mercury_7040740`. Each configuration runs four identical native N forwards, with full FP32 model, eval, cache off, TF32 off, torch profiler, actual loop/head counters, checkpoint/source hashes and weight-version guards. Source/fixture permission receipts are written before model loading. Source-level existing forward assertions remain intact; cross-repeat comparisons record differences rather than conceal them.

| Configuration | Actual profiler SDPA operator / forward | Q9 K1 Q equal pairs / 6 | K1 full-R-logit equal pairs / 6 | K2 full-R-logit equal pairs / 6 |
|---|---|---:|---:|---:|
| Original FP32 runtime | 48 efficient-attention dispatches | 0 | 0 | 0 |
| New deterministic + explicit math runtime | 48 math-attention dispatches | 6 | 6 | 6 |

Original all-pair maximum absolute differences are respectively `0.19222480058670044`, `0.6823883056640625`, `0.6892385482788086`. Maximum non-equal element counts are 137216 for Q9 K1 Q and 49152 for both endpoints. All three tensors have zero differing elements and exact zero maximum difference in every math-run pair. This directly reproduces the donor identity failure in native-only forwards before any P intervention; it is a technical reproducibility failure, not a negative scientific result.

The second configuration is a separate process that sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before torch import/CUDA, uses `torch.use_deterministic_algorithms(True)`, explicitly scopes every forward with `SDPBackend.MATH`, and disables TF32. Several runtime controls change together: this observation identifies a stable combined configuration on the one fixed technical question; it does not isolate efficient SDPA as the sole cause, establish all-eight stability, or establish a pure dtype effect.

Server raw output roots are `/root/autodl-tmp/ouro-k1-k2-mechanism/numeric_fp32_native_repeat_original_v1/` and `numeric_fp32_native_repeat_deterministic_math_v1/`. Each contains four gzip safetensors archives with actual Q9 K1 Q and K1/K2 full-vocabulary R logits, per-repeat JSON with dispatch/count/hash records, source/permission freeze, runtime identity and COMPLETE with all six pair comparisons. Logs are adjacent `<directory>.log`. Small COMPLETE and first-repeat metadata copies are retained locally as `FP32_NATIVE_REPEAT_*_{COMPLETE,FIRST}.json`.

Prepared, but not executed here: `C1_FP32_NUMERICAL_MATH_v1.py`, SHA `ca8b20eca73e6d69d5a8dc553bc60918af4e9ee7af8e4be31015531ff847ec3d`, separate output `numeric_fp32_deterministic_math_v1`. It retains the original fixed eight IDs, fresh FP32 N/P differences and four-corner conditions, all donor/branch/endpoint exact assertions, actual dispatch profiler and BF16 native-source bridge guards. Only the explicitly documented deterministic/math configuration and namespace change. Root must review and separately launch this full-eight backend-sensitivity run; no assertion tolerance was relaxed.
