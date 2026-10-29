# Next bounded hypothesis: calibrate an input response into intervention value

Status: specified after the 400-question analysis; not executed in this directory. This is not a report of a successful Quality Head.

The observed input-loss response S is measured in nats/token. For full rollback it was negative on all 385 probe-available questions, although the directionally fixed response had an exploratory AUC of 0.769 on 10 beneficial and 13 harmful events. The current sign gate therefore never chose full rollback. This motivates testing calibration, not selecting a negative threshold from these outcomes.

Use the already separated 3,200-question head-training manifest to collect actual frozen-state intervention outcomes under the unchanged depth-2 action family. Use the 800-question validation manifest for model screening, not training. Neither set is a fresh independent benchmark: both were used in the first research round with different actions. They remain development data. The 400 discovery questions and eight-question pilot never enter fitting or calibration selection.

For each nonzero action a, the target is `1[pred_final_after_state_action_a=gold] - 1[pred_final_Adaptive=gold]`. Include neutral examples. At this early decision point future answer agreement is unavailable, so do not use the previous retrospective agreement mask. Gold is used only to form training/evaluation targets after the forward passes.

Exactly three fixed linear calibrators, no hyperparameter search:

- Input-response head: two features per action, L_A(0)-L_A(a) and L_B(0)-L_B(a), plus intercept: 9 scalar fitted parameters across three actions.
- Confidence control: full-vocabulary margin change and negative entropy change at the same decision point, plus intercept: also 9 fitted parameters. Signs are learned from training, not flipped on evaluation results.
- Combined ablation: the same four features plus intercept: 15 fitted parameters.

Standardize each feature using only the probe-available training rows, Ridge alpha=100. Rows lacking the required probes use Adaptive for every learned policy and are excluded from fitting these paired comparison heads. Predictions are action-specific expected accuracy changes relative to Adaptive; continue is assigned value 0. Select the maximum positive predicted gain, otherwise continue. Action order is [continue, rollback, partial rollback, extrapolate]; no threshold tuning or abstention search. A positive fitted value is not a statistical guarantee for an individual question.

Compare against a training-mean constant-action predictor, the unchanged training-free probe rule, confidence rule, fixed state actions, matched-action-frequency shuffling and oracle. Record whether the calibrated input-response head adds value beyond equally calibrated confidence; the raw 0.331 margin AUC cannot be used as a substitute for that comparison. Selection of a deployable head must be frozen after validation and before any new independent final task. No old final dataset is reused for confirmation.

Bound counterfactual collection, fitting and validation to at most 3 additional GPU hours, including failed processes, within the remaining original allowance (about 5.93 hours after this study). Measure actual throughput before committing the whole collection and reserve budget for independent confirmation. A capacity estimate is not a promise of success. If the action-value signal or controller fails validation, retain that result and do not expand the head or tune evaluation thresholds.
