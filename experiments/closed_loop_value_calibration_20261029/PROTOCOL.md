# Frozen-state action-value calibration

This is the bounded supervised follow-up specified in `closed_loop_action_value_dev400_20261029/NEXT_CALIBRATION.md` at public commit e13c487bd709abb8d3bb410456f39a7b1eb2be28. The 400-question training-free controller failed; its rollback response supplied only an exploratory ranking hypothesis. No success or novelty is assumed here.

## Data and intervention

Use the original disjoint OpenBookQA head_train (3,200) and validation (800) manifests, with their existing official zero-shot prompts and full-candidate character-normalized scoring. Both were exposed in the first research round with different interventions; validation is held out from this head fit, not a fresh benchmark. Discovery400, ARC1172, Qwen600, SciQ1000 and PIQA1802 are excluded from fitting and from new final confirmation.

Import the sealed corrected state-intervention functions from the 400-question study. At depth 2, apply h2+a*(h2-h1), a=[0,-1,-0.5,+0.5], then execute the remaining native loops and Adaptive LoopCD readout. The same frozen Ouro-1.4B revision, BF16, no cache and no BOS apply. Input-probe controls and all action counterfactuals are recorded before attaching gold. Preserve exact input-question and candidate formatting; baseline score parity stays <1e-5 absolute.

## Targets and heads

The target for each nonzero action is its final accuracy indicator minus the Adaptive accuracy indicator. Use all probe-available training rows, including neutral targets; no future answer-agreement filter. If either beneficial or harmful training events are fewer than 20 for any action, stop before fitting and report insufficient supervision rather than claiming a calibrated three-action controller.

Three fixed Ridge(alpha=100) models, independently fitted for each action:

- `input`: gains in the two known-input token probe groups A and B, 2 features/action and intercept, 9 fitted linear coefficients total.
- `confidence`: intermediate full-vocabulary margin change and negative entropy change, 2 features/action and intercept, also 9 coefficients. Training may learn signs; no evaluation-label sign flips.
- `combined`: all four features/action and intercept, 15 coefficients.

Store training means/scales separately from these coefficient counts. Standardize using only probe-available training rows; constant feature scale is set to 1. Missing required probes forces Adaptive for every learned policy. No activation clipping, architecture search or learned backbone parameter. All features are constructed from an observation object containing only `decision` and `controls`; action outcomes and gold are unavailable to that interface.

Continue has value 0. Choose the maximum predicted positive gain; otherwise continue. Ties use [continue, rollback, partial rollback, extrapolate]. Threshold is exactly 0, fixed before collecting these labels. A positive regression output is not a guarantee or calibrated confidence bound for an individual example.

Training also freezes a probe-available mean-gain constant predictor and the best fixed action over all training questions. No discovery data enter these choices. Save model/source/train hashes in TRAIN_LOCK before validation starts.

## Validation and selection

Report all three heads, Adaptive, all fixed actions, unchanged training-free probe/confidence rules, training-mean predictor, terminal-only selected-action ablations, and oracle. Apply exact paired McNemar and 10,000 paired bootstrap resamples. For each method candidate (`input`, `combined`), preregister four primary tests: versus Adaptive, versus the calibrated `confidence` head, versus the training-selected best fixed action, and versus 10,000 action-frequency-preserving shuffles among probe-available validation questions. Holm adjusts these eight primary p values jointly.

Candidate admission requires net >=5/800 and positive lower 95% paired CI in all three paired comparisons, and adjusted p<0.05 for all four primary tests. If both pass, select the smaller `input` head; `confidence` is a control, not evidence of information beyond confidence. If neither passes, retain Adaptive and do not open a new final benchmark for this candidate family. All other comparisons are explicitly secondary. To claim propagation is necessary, additionally report the paired terminal-only ablation; it is not an automatic consequence of a selected state policy.

On nonzero per-action accuracy-change events, report AUC, class counts and 2,000 stratified-bootstrap intervals for each head and the raw input response; no sign correction. Report all-row squared error against accuracy delta and training-frozen quintile calibration bins (boundary quantiles computed from training predictions). Ranking or reduced regression error does not substitute for an accuracy gain.

## Budget and execution

Total new GPU-worker allowance: 10,800 seconds, counting failures and any repeated collection, within the remaining original 21,357.79 seconds. The supervisor also bounds its complete wall-clock chain at 10,800 seconds. Training collection has an 8,500-second maximum to reserve validation capacity. After the first 16 training rows, measure per-question throughput and require projected collection of all remaining training and validation rows with a 10% reserve to fit within the global deadline. These 16 rows remain part of training; no extra data pilot or outcome-based pruning.

Collect training, fit the locked heads with CPU, write TRAIN_LOCK, then collect validation with the same worker functions. Fit does not load validation outcomes. Any failure stops the chain and preserves process receipts and partial data; no automatic retry or reset. A future independent task requires a separate final lock and equal-compute comparisons. No unrelated experiment is resumed.
