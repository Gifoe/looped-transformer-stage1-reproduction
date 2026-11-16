# Final scientific decision

**Principal recommendation: D. Stop CATC.**

Author: Sol. The requested ASTRA filename is retained for compatibility; no Astra agent was used.

The chosen fixed policy is P- with coefficients [0, -0.1]. It was selected on development candidate CE and frozen before new control/Validation outcomes. On development K4, native and the gold best-of-five accuracy oracle both score 229/256; this intervention set offers no additional correct answer even to the offline oracle. Direction-dependent CE shifts exist but are small and do not establish a useful controller.

On 511 clean current confirmation questions, selected CATC gets 446, native K4 446, matched random 449, and matched scaling 446. The primary paired accuracy gate fails. The separate CE-specificity gate is False; any score-only specificity remains separate from the requested accuracy goal. Actual BF16 doses, including unmatched sites, are retained.

On the remaining 326 confirmation questions, selected/native K4 get 291/290. On exposed Validation299, they get 261/261; E6 conservative gets 263. Final K2 and K4 tables, paired intervals, damage and all-native-loops-wrong corrections remain visible.

A common reliable training-free accuracy gain is not established, so A is unsupported. The previous bounded static 14-scalar fit selected the zero identity checkpoint and did not improve independent SciQ; this study provides no evidence that layer-specific fitting is necessary, so B is unsupported. The development accuracy oracle has no headroom, and no label-free predictor was tested; C is not justified by heterogeneous score directions or a CE oracle. No conditional-controller training is authorized or launched.

This stops the tested CATC development route under the fixed checkpoint/protocol/coefficient set and present evidence. It does not prove that every conceivable information-flow intervention is useless. Algebraic Value/Routing components are convention-dependent, and repeated Train/Validation exposure limits inferential independence. The corrected511 excludes one exact development-prompt duplicate found after initial aggregates; original512 remains visible and no policy was reselected. Completing the bounded experiments is not achievement of the overarching objective of outperforming native K4 and strong output-only baselines.
