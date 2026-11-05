# R33 pre-effect numerical domain

This additive CPU implementation contract records the Root/Astra decision without changing the original design. No model or new head has run.

The FP64 Q-native probability vector must be finite and strictly positive at every vocabulary coordinate. Softmax underflow to zero is numeric-domain invalid. There is no probability floor, alternate metric or epsilon repair.

Every main/LOO centered correction must have a finite weighted squared norm, and `(norm2 == 0)` must equal `all(h == 0)`. A nonzero correction whose weighted norm underflows to zero is invalid, even if all probabilities are positive. Exact-zero corrections return the original literal FP32 Fixed tensor in both the primary and matched controls. Nonzero corrections require a finite positive norm.

Generic, vocabulary-null and reference-consensus direction degeneracy uses the unchanged 64-epsilon rule. A nonzero main correction with a degenerate control direction is unmatched, precluding both effect gates without adding an epsilon. A zero guidance norm with nonzero main correction likewise makes the matched-scale control unmatched. All remaining intermediates/scales/outputs must be finite.

The signed-min residual interval theorem concerns unnormalized gain coordinates. Centering and full-vocabulary normalization can reverse a probability-change sign; the frozen CPU counterexample must demonstrate this. No do-no-harm or correctness theorem is asserted.

The raw resource receipt `r33_RESOURCE_CACHE_PREFLIGHT.json` is a private audit artifact and must be excluded from publication, required public inputs and archives. The separate `r33_RESOURCE_CACHE_PREFLIGHT_PUBLIC.json` retains aggregate ABI/cost/time evidence while omitting host/GPU identifiers and user home paths.
