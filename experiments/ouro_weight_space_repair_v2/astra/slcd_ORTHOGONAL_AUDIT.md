# SLCD logit decomposition and scoring audit

## Primary-source finding

Checked [LoopCD v1, Section 5.2](https://arxiv.org/html/2610.02185v1#S5.SS2) and [Appendix F.2](https://arxiv.org/html/2610.02185v1#A6.SS2). They describe decomposing the early-to-final logit contrast into components parallel and orthogonal to the final logits. They do **not** explicitly define vocabulary-mean centering or give the projection quotient. The inspected section math elements contain z_R-z_1 and z_R, not a centering operator. F.2 discusses an empirical near-equivalence with hidden-state guidance; it does not justify claiming exact identity of the two algorithms.

Accordingly, `centered=True` below is an explicit implementation convention for additive-logit-shift invariance. It is not attributed to an unspecified author implementation. `centered=False` is the literal raw-vector projection control. They require separate recipe IDs and must not be silently pooled as one reproduction.

## Defined formulas

Each vector is one target-token prediction position over the **full vocabulary**. Define C(x)=x-mean_v(x), independently at each position. In centered mode:

```
a = C(z_final)
d = C(z_final - z_reference)
parallel   = a * dot(d,a) / dot(a,a)
orthogonal = d - parallel
```

Raw mode uses a=z_final and d=z_final-z_reference. If dot(a,a) is exactly zero, the projection onto its zero-dimensional span is defined as parallel=0 and orthogonal=d. No epsilon ridge is added to a nonzero denominator. All arithmetic is FP32; inputs must already be FP32 casts of BF16 lm_head outputs. Overflow/nonfinite values are rejected. Both inputs and result components are position-specific; no pooling over tokens/candidates is performed.

Centering is not necessary to calculate a raw Euclidean projection. It is necessary for this implementation's probability-level invariance to independent arbitrary additive constants in final/reference logits: the uncentered projection coefficient otherwise depends on the chosen logit origin. This is our mathematical rationale, not a documented detail of the cited paper. Centered components sum to the centered contrast, whose difference from the raw contrast is a softmax-invisible constant in exact arithmetic. Floating-point shift invariance is tested within tolerance, not asserted bit-exact.

Readout definitions in `scripts/slcd_logit_methods.py`:

```
fixed     = z_final + omega*(z_final-z_reference)
adaptive  = z_final + omega*(1-top2_probability_gap(z_final))*(z_final-z_reference)
two_sided = fixed + gamma*(z_final-z_late)
orthogonal_readout = z_final + omega*orthogonal
parallel_readout   = z_final + omega*parallel
```

The late term SUBTRACTS the late update. Orthogonal/parallel are the single-sided early-contrast ablations; they do not filter a late residual. Gamma is accepted only for two_sided_logit (zero is harmless on other kinds). Gamma0 takes the identical fixed arithmetic path and does not require late logits. It does not mean orthogonal/parallel become Fixed. The parallel readout acts as a scalar rescaling plus a constant in centered mode; token order is preserved only when the resulting scale is positive. No coefficient clipping is silently inserted.

Adaptive omega means its cap: use omega=1.0 for the existing cap1 baseline. Fixed and projection defaults are omega=.5. Index defaults are reference1/final4/late5. Explicit different references and late6 are supported without supplying a search grid. Strengths are bounded to [0,1]; the controller, not this module, freezes actual candidate values.

## Scoring API and limits

`score_logit_recipes(logits, targets, recipes)` returns `{recipe_id: float(sum_loglikelihood)}`. `logits[k]` has shape [all continuation prediction positions, vocabulary]; `targets` contains one integer token ID per position. Every recipe first constructs full-vocabulary logits, then applies log_softmax across vocabulary, gathers all target tokens and sums in FP32. Character normalization is performed later by the caller. Score-level linear combinations are not a substitute because each token's log-partition term changes. IDs must be unique, input shapes/dtypes finite and consistent, and inputs detached. Functions use no_grad and never mutate logits, load models, edit weights, open data, or select recipes.

## CPU synthetic verification

`scripts/slcd_test_logit_methods.py` and `astra/slcd_SYNTHETIC_TEST.json` verify seven target positions and vocabulary19, with synthetic BF16-rounded FP32 logits. Fixed and two-sided gamma0 scores are exactly equal; adaptive cap1 matches the existing formula exactly. A positive late-only token update is subtracted, not added. Centered orthogonality max absolute dot error was 1.78814e-6. Independent per-token/depth additive shifts changed centered orthogonal/parallel log probabilities by at most 1.43052e-6; raw controls changed by over1.15, confirming the distinction. The full-logit two-sided score differed from an invalid score-level shortcut by10.03823 in this synthetic example. Zero-norm projection, alternate reference/late indexing, input immutability and invalid recipe rejection passed.

No local/server ARC test files, candidate test scores, historical result artifacts, model weights, or GPU experiments were accessed for implementation/testing. A web-tool find response automatically included an adjacent F.1 illustrative ARC passage; it was excluded from method design and tests. Subsequent HTML extraction selected only the requested S5.SS2/A6.SS2 math elements and terminology counts. No recipe was chosen using that adjacent material. Only new `slcd_` files were added in existing scripts/astra directories. These tests validate arithmetic; they provide no task-accuracy evidence.
