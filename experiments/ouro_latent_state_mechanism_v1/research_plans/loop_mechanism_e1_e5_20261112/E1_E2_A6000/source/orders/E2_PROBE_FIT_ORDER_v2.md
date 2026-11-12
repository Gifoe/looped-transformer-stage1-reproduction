# E2 probe fit order

UTC 2026-11-12T13:36:09.837384+00:00

Train non-demo 1094; validation299 chooses lambda; test untouched until model freeze. Two fixed R sites L18 layer-input and post-norm; loops K1/K2 trained separately, cross-evaluated. Raw and unit-L2 features use a separate pooled K1+K2 train-only unlabeled shared mean/std per site/type; zero std becomes one. Candidate target-ID classes from train only; whole example unsupported for any OOV candidate, first tie in original candidate order. Masked multiclass mean cross-entropy plus 0.5*lambda*sum(weight squared); intercept unpenalized. L2 {1e-4,1e-2,1,1e2,1e4}, zero init, SciPy L-BFGS-B maxiter800 ftol1e-9 gtol1e-6; require finite converged fits. Select same-loop validation cross-entropy; ties choose larger lambda. Fixed seed 20261112 permutes training labels within identical candidate sets; constant uniform, global letter and candidate-count-conditioned letter priors as controls. Save all candidate logits/probabilities/predictions/ties, selected weights/transforms and source hashes; freeze before any test evaluation.


## Uniform iteration-budget amendment
UTC 2026-11-12T13:40:55.399301+00:00

First attempt stopped before model selection or any test evaluation: l18_input raw K2 permuted lambda=1e-4, STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT at configured 800. Exact nfev, loss, and gradient infinity norm were not saved; unknown, not inferred. Preserve first attempt and its journal. All 80 fits restart with maxiter=4000 uniformly, identical loss/gradient, regularization grid, ftol/gtol, zero init, seed, train/validation data, transform and selection rule. If any fit still fails, stop and investigate; no further mechanical increase or removal of a lambda. Output uses independent probe_fit_attempt2.
