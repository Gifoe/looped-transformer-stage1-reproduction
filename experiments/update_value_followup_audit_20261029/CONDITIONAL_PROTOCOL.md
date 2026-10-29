# Bounded development-only conditional-value check

This is a follow-up to the failed first-round controller, under the continuing goal. It does not revise the original protocol, results or No-Go decision. SciQ and PIQA are already exposed and will not be scored with the new heads. The existing 800-question validation set is reused as development evidence, not independent confirmation.

For accuracy gain, equal predicted answers certify zero gain without labels. Thus

`E[delta_accuracy | x,a] = I[pred_a != pred_Adaptive] * E[delta_accuracy | x,a,pred_a != pred_Adaptive]`.

The recorded training data have 8686 same-answer pairs among9600 pairs. Removing these certified-zero pairs from the conditional regression changes the target population without introducing a bigger classifier or additional features. This is a standard conditioning identity, not a novelty claim.

Exactly two candidate heads: the existing new-signal features, and the existing confidence-control features. For each action, fit Ridge(alpha=100) only on its training answer-disagreement rows; standardize using those training rows only. Keep both-wrong disagreements with target0. No backbone training. Parameter counts remain345 and96 respectively, matching their original heads. The inference value is exactly0 for same-answer actions, and the conditional prediction otherwise. Retain Adaptive unless the maximum value exceeds0.05; deterministic action order Keep, Forward, Rollback. No sign, alpha, threshold, feature or architecture search.

Only OpenBookQA head_train3200 and validation800 can enter this check. Record predictions and paired accuracy, corrections/degradations, exact McNemar and paired bootstrap. Apply the original numerical gate (net>=5, positive lower95% CI, Holm p<.05), conservatively including the original11 policy p values plus these2. Passing is only a reason to define a genuinely new confirmation experiment; reusing validation cannot provide confirmatory significance. Failure is a reason not to spend GPU time on this candidate family.

No new GPU work and no new final-data acquisition is part of this check. Freeze this protocol and the script hash before fitting. A negative result must remain negative; do not respond by changing settings.
