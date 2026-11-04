# Follow-up target: at least733/1172 (62.5%)

Current verified ARC test remains **726/1172**. The target has **not** been reached. No follow-up method has been evaluated on ARC test yet. On2026-11-04 connectivity was restored; R5 completed and R6 has now been frozen for one full test.

The original A1 method/lock and all older V2 files are preserved. The user's higher target initiated further training-only research in this same folder. All1093 train questions and official validation had already been used, so deterministic5-fold slices are descriptive robustness checks, not independent validation.

| Follow-up family | Best train correct/1093 | Incumbent | Decision |
|---|---:|---:|---|
| R2 Fisher late-residual decorrelation, Euclidean/matched-norm/cosine controls |689|690|Rejected; no test|
| R3 four candidate-score compositions |691|690|Only+1; not tested|
| R4 Fisher energy/exact-KL/uniform energy budgets and Adaptive+late control |687|690|Rejected; no test|
| R5 fixed answer-prior corrections |691|690|All6 failed the+7 training gate; no test|
| R6 partial/adaptive prior correction |705|690|Projection selected; complete test pending|

R6 selected `r6_prior_projection` on reused train1093:22 rescues/7 regressions vsA1, net+15. Gains were+6/+1/+8 across discovery384/selection192/confirmation517 and+2/+3/+2/+4/+4 across the five fixed robustness slices. Matched-magnitude control scored692, half-strength698, quarter-strength690. These are development results after adaptive research, not independent evidence or a guarantee of733 on test.

The selected score is A1-alpha*e, where e=Fixed_prior-K4_prior, g=Fixed_ctx-K4_ctx, and alpha=clip(dot(center(g),center(e))/dot(center(e),center(e)),0,1), with zero denominator giving0. Prior uses a fixed `Answer:` prefix and complete candidate likelihoods; original contextual scores retain exactly the same25-shot prompts. No gold enters the rule. The original unedited model, environment and driver were rechecked after connection recovery.

The R6 final lock SHA is `6de605b885b4fb9ff72b273ecc0c0484e9fe72fc5d51d793b22367db879b7da7`. The lock includes selected coefficients/ranking source, original model/tokenizer identity, data/prompt hashes, logical context/prior depths5/4, and authorization before prior test collection. Only this primary will be tested; no test-based candidate replacement is allowed.

R4/R5's predeclared test guard requires at least+7 correct on train1093 vsA1 and at least3/5 nonnegative folds, before one primary can be frozen. This does not guarantee seven additional test corrections. No coefficients are chosen from new test results.

Full HellaSwag for original A1 was intentionally yielded after1956 saved rows to prioritize the user's higher accuracy target. Those rows and all worker time remain preserved; the partial sample is not reported as full benchmark accuracy. Initial network download failures were also preserved and charged. The independent additional6h GPU-worker ledger remains in force.

The previous SSH blocker was resolved on2026-11-04. The6h GPU-worker ledger continues from its existing balance; no budget reset occurred. Full R6 test results and conditional cross-task results will be added when verified.

Research claims are limited: projection guidance, Fisher logit geometry, KL trust regions and PMI calibration have prior art. See astra/slcd_FOLLOWUP_RELATED_WORK.md. No novelty or causal mechanism is inferred from these negative training comparisons.
