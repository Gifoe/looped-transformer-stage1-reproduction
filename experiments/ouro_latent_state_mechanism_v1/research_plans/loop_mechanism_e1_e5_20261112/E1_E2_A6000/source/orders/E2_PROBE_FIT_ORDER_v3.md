# E2 probe fit order

UTC 2026-11-12T13:36:09.837384+00:00

Train non-demo 1094; validation299 chooses lambda; test untouched until model freeze. Two fixed R sites L18 layer-input and post-norm; loops K1/K2 trained separately, cross-evaluated. Raw and unit-L2 features use a separate pooled K1+K2 train-only unlabeled shared mean/std per site/type; zero std becomes one. Candidate target-ID classes from train only; whole example unsupported for any OOV candidate, first tie in original candidate order. Masked multiclass mean cross-entropy plus 0.5*lambda*sum(weight squared); intercept unpenalized. L2 {1e-4,1e-2,1,1e2,1e4}, zero init, SciPy L-BFGS-B maxiter800 ftol1e-9 gtol1e-6; require finite converged fits. Select same-loop validation cross-entropy; ties choose larger lambda. Fixed seed 20261112 permutes training labels within identical candidate sets; constant uniform, global letter and candidate-count-conditioned letter priors as controls. Save all candidate logits/probabilities/predictions/ties, selected weights/transforms and source hashes; freeze before any test evaluation.


## Uniform iteration-budget amendment
UTC 2026-11-12T13:40:55.399301+00:00

First attempt stopped before model selection or any test evaluation: l18_input raw K2 permuted lambda=1e-4, STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT at configured 800. Exact nfev, loss, and gradient infinity norm were not saved; unknown, not inferred. Preserve first attempt and its journal. All 80 fits restart with maxiter=4000 uniformly, identical loss/gradient, regularization grid, ftol/gtol, zero init, seed, train/validation data, transform and selection rule. If any fit still fails, stop and investigate; no further mechanical increase or removal of a lambda. Output uses independent probe_fit_attempt2.


## Objective correction (all candidates retained)
UTC 2026-11-12T13:42:41.660924+00:00

Training non-demo1094 candidate occurrences {330:1094,340:1094,389:1094,414:1,422:1093}; gold counts {330:232,340:287,389:286,414:0,422:289}. Class414 has no positive gold, so an unpenalized class bias can tend to negative infinity with no finite optimum. This mathematical defect is not asserted to be the sole cause of the first failed fit. To ensure a finite well-defined objective, regularize ALL coefficients including bias by the same lambda. Restart all 80 fits under the same five lambdas and 4000-iteration budget, same transform/seed/solver/tolerance and validation selection; no class or question dropped. Attempt2 logged partial validation selection before stopping, and is not an unseen validation result; no complete selected model or test evaluation occurred. Selection and test usage remain separated; earlier ARC outcomes already exposed. Preserve both previous attempts and logs.
