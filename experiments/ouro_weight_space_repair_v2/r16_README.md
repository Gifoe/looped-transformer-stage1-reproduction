# R16: bounded native-KL approximation audit

This is one post-evaluation development pilot, not a new final-ARC primary. HellaSwag/OpenBookQA and other external tasks remain paused. All work stays in the existing experiment directory.

R15 found no useful rescue/damage association for its context-preference signal. That failure does not establish that Fisher approximation error caused earlier failures. R16 tests that narrower possibility once.

For each current continuation prefix, keep native `p0=softmax(Q4)` fixed. Let `F=Q4+.5*(Q4-Q1)` and `e=center_p0(.5*(N0_4-N0_1))`. Choose `a in [0,1]` to minimize `KL(p0||softmax(F-a*e))`; solve the convex one-dimensional problem by endpoint derivative tests or 32 fixed FP64 bisections. The generic helper includes reference dispersion for mathematical checks, but the empirical pilot uses only N0, J=1, U=0.

This is standard exponential-family KL calibration, not a correctness objective. Reducing its objective can worsen the answer. FP64 objective checks and explicit FP32 rounding bounds are separate. Full-vocabulary probabilities are actually evaluated; candidate likelihoods are not treated as vocabulary logits. Final full-target likelihood, character denominator, late Q4-Q5 term, fixed 25 demonstrations, original checkpoint and original attention path are preserved. No training, weight modification, KV optimization, learned gate or coefficient search.

The fixed uniform64 development sample is drawn using seed2021 without labels/predictions. Gold is used only after the complete worker and all parity checks pass. The primary must strictly beat R6, old M2 and its own two controls and retain every pilot R6-only-vs-Fixed success. The norm control permits rho>1; the shuffle preserves correction norm. Controls are ineligible for promotion.

One 120-second worker plus at most15 seconds termination grace inherits all five prior GPU ledgers under the unchanged21600-second aggregate cap. Any failure or partial completion is charged and blocks the analysis. No automatic retry. Passing would authorize a cost/experimental-scope review only; it would not authorize new test or other benchmarks. Failure ends this objective without strength/sign/J3 retuning.

Synthetic solver checks: `python -B scripts/r16_math_checks.py`. Prepare once: `python -B scripts/r16_prepare.py`. The definition and sources must be public and verified before a worker can start. Linux launcher: `env/bin/python -B -u scripts/r16_launch.py`.
