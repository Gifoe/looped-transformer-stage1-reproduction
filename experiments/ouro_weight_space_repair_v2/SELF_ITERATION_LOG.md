# Training-free LoopCD self-iteration

Development uses only train discovery384 / selection192. Official validation is reserved for one final selection stage. Historical test726 is not selection evidence.

## Iteration 1

Hypothesis: Late update disagreement may correct harmful Fixed extrapolation at the complete-candidate score level.

Motivating evidence: Original training scores exist for384 discovery and192 independent selection; no new GPU forward required.

Modification/configuration: A1=Fixed+gamma*(K4-K5), gamma=[0,.25,.5,.75,1]; complete-candidate character normalization; gamma0 exact control.

Observed development results: {"discovery": {"native4": {"correct": 239, "corrections": 14, "regressions": 10, "mean_delta_margin": -0.031568841609552555}, "fixed": {"correct": 235, "corrections": 0, "regressions": 0, "mean_delta_margin": 0.0}, "adaptive": {"correct": 240, "corrections": 8, "regressions": 3, "mean_delta_margin": 0.0027265095799105243}, "A1_g0p0": {"correct": 235, "corrections": 0, "regressions": 0, "mean_delta_margin": 0.0}, "A1_g0p25": {"correct": 238, "corrections": 3, "regressions": 0, "mean_delta_margin": 0.0021873714953152185}, "A1_g0p5": {"correct": 236, "corrections": 5, "regressions": 4, "mean_delta_margin": 0.004337419194200414}, "A1_g0p75": {"correct": 235, "corrections": 7, "regressions": 7, "mean_delta_margin": 0.006281038077885072}, "A1_g1p0": {"correct": 237, "corrections": 11, "regressions": 9, "mean_delta_margin": 0.008038715089067406}}, "selection": {"native4": {"correct": 114, "corrections": 4, "regressions": 4, "mean_delta_margin": -0.039072785446846114}, "fixed": {"correct": 114, "corrections": 0, "regressions": 0, "mean_delta_margin": 0.0}, "adaptive": {"correct": 113, "corrections": 2, "regressions": 3, "mean_delta_margin": -0.007247875689335458}, "A1_g0p0": {"correct": 114, "corrections": 0, "regressions": 0, "mean_delta_margin": 0.0}, "A1_g0p25": {"correct": 115, "corrections": 1, "regressions": 0, "mean_delta_margin": 0.0028012770678135317}, "A1_g0p5": {"correct": 114, "corrections": 1, "regressions": 1, "mean_delta_margin": 0.005458750901659897}, "A1_g0p75": {"correct": 115, "corrections": 2, "regressions": 1, "mean_delta_margin": 0.007933989344502066}, "A1_g1p0": {"correct": 116, "corrections": 3, "regressions": 1, "mean_delta_margin": 0.010324107385188326}}}

Decision: Retain gamma.25 and1 as distinct tradeoffs. Discovery gains3/2, selection gains1/2; stronger gamma raises average margin but also regressions. No statistically reliable development improvement established.

Next action: Test token-logit correction and centered/raw orthogonal guidance using original-model saved states; compare full log-softmax likelihoods. Do not inspect validation/test.

## Iteration 2

Hypothesis: Token correction and removal of parallel temperature-like guidance may preserve useful ranking changes. Candidate-score fragility may identify avoidable flips.

Motivating evidence: A1 gains were small and margin increased while accuracy was nonmonotonic. Native-to-Fixed train feature audit showed avoidable low-margin changes.

Modification/configuration: 15 complete-token recipes incl A2 gamma grid, centered/raw orthogonal three strengths and parallel controls; 8 deterministic candidate-aware pilot rules.

Observed development results: {"discovery": {"tf_fixed": 0, "tf_adaptive": 5, "A2_g0p0": 0, "A2_g0p25": 5, "A2_g0p5": 1, "A2_g0p75": 1, "A2_g1p0": 4, "O_centered_w0p25": 7, "O_centered_w0p5": 8, "O_centered_w1p0": 1, "P_centered_w0p5": -5, "O_raw_w0p25": 5, "O_raw_w0p5": 6, "O_raw_w1p0": 2, "P_raw_w0p5": -4, "C_agree45_revert": 7, "C_support5_only": 7, "C_native_hi_revert_t0p01": 4, "C_fixed_low_revert_t0p01": 3, "C_native_hi_revert_t0p025": 3, "C_fixed_low_revert_t0p025": 7, "C_native_hi_revert_t0p05": 2, "C_fixed_low_revert_t0p05": 7}, "selection": {"tf_fixed": 0, "tf_adaptive": -1, "A2_g0p0": 0, "A2_g0p25": 1, "A2_g0p5": 1, "A2_g0p75": 1, "A2_g1p0": 2, "O_centered_w0p25": -2, "O_centered_w0p5": -2, "O_centered_w1p0": -5, "P_centered_w0p5": 5, "O_raw_w0p25": -2, "O_raw_w0p5": -3, "O_raw_w1p0": -4, "P_raw_w0p5": 2, "C_agree45_revert": 1, "C_support5_only": 1, "C_native_hi_revert_t0p01": 0, "C_fixed_low_revert_t0p01": 0, "C_native_hi_revert_t0p025": -1, "C_fixed_low_revert_t0p025": 1, "C_native_hi_revert_t0p05": 0, "C_fixed_low_revert_t0p05": 0}}

Decision: Retain A2 gamma.25 and1, and candidate low-Fixed-margin/native-consensus protection. Abandon pure orthogonal/parallel as standalone primary branches: signs reverse across discovery/selection.

Next action: Combine low-strength token correction with a single protective cue; test neighboring gamma values to assess narrow-interval sensitivity before final validation.

## Iteration 3

Hypothesis: Late correction can be retained while reverting fragile or unsupported contrastive choices to native K4.

Motivating evidence: Train-only failures concentrate among low-margin Fixed changes; original Fixed choices supported by K5 have a higher rescue/damage ratio. A2 gamma.25 had5/1 net gain with no regressions.

Modification/configuration: 10 bounded hybrids: A2 gamma.25 or A1 gamma1, one confidence threshold [.01,.025,.05] or one K5 support/disagreement rule. No gold-dependent inference.

Observed development results: {"discovery": {"H_A2_g0p25_protect0p01": 4, "H_A2_g0p25_protect0p025": 7, "H_A2_g0p25_protect0p05": 7, "H_A2_g0p25_k4_k5_disagree": 7, "H_A2_g0p25_cd_agrees_k5": 11, "H_A1_g1p0_protect0p01": 2, "H_A1_g1p0_protect0p025": 11, "H_A1_g1p0_protect0p05": 8, "H_A1_g1p0_k4_k5_disagree": 5, "H_A1_g1p0_cd_agrees_k5": 7}, "selection": {"H_A2_g0p25_protect0p01": 0, "H_A2_g0p25_protect0p025": 1, "H_A2_g0p25_protect0p05": 0, "H_A2_g0p25_k4_k5_disagree": 1, "H_A2_g0p25_cd_agrees_k5": 2, "H_A1_g1p0_protect0p01": 2, "H_A1_g1p0_protect0p025": 2, "H_A1_g1p0_protect0p05": 1, "H_A1_g1p0_k4_k5_disagree": 1, "H_A1_g1p0_cd_agrees_k5": 3}}

Decision: Retain H_A1_g1p0_protect0p025 and H_A2_g0p25_cd_agrees_k5 (+11/+2 on384/192), along with simple A2.25 and A1.25/1 controls. Other combinations do not consistently improve selection.

Next action: Check gamma neighbors and closer early references / weaker full guidance; then confirm surviving hypotheses on previously unused ARC training examples before official validation.

## Iteration 4

Hypothesis: A small late correction should remain useful near gamma.25; a closer early reference may reduce overshoot without an additional loop.

Motivating evidence: Pure orthogonal effects reverse between train subsets; A2.25 has zero regressions but its strength curve is nonmonotonic. Hybrid protection improves both subsets.

Modification/configuration: Six bounded additions: gamma.125/.375, full guidance omega.25/.75, referenceK2/K3 atomega.5.

Observed development results: {"discovery": {"A2_g0p125": 2, "A2_g0p375": 3, "F_w0p25": 6, "F_w0p75": 3, "Ref2_w0p5": 3, "Ref3_w0p5": 2}, "selection": {"A2_g0p125": 1, "A2_g0p375": 1, "F_w0p25": 0, "F_w0p75": -1, "Ref2_w0p5": 2, "Ref3_w0p5": 1}}

Decision: Small gamma neighbors retain selection+1 but discovery gains shrink2/3 vs5 at.25. ReferenceK2 gives+3/+2 with no extra loop; retain as cheap candidate. Weak/strong full guidance and referenceK3 are weaker.

Next action: Freeze eight provisional survivors and check them on517 previously unused official train examples; no validation/test access.

## Iteration 5

Hypothesis: Earlier hybrid gains may be sample-specific; transfer must survive unused official train questions.

Motivating evidence: Selection gains only2-3 questions and substantial searched candidate pool.

Modification/configuration: Freeze8 survivors before fresh517 train feedback; advance top3 by predeclared correct/depth/complexity/ID; no new method or threshold.

Observed development results: [{"method": "native4", "correct": 325, "corrections": 12, "regressions": 20, "net_correct": -8}, {"method": "fixed", "correct": 333, "corrections": 0, "regressions": 0, "net_correct": 0}, {"method": "adaptive", "correct": 335, "corrections": 7, "regressions": 5, "net_correct": 2}, {"method": "H_A1_g1p0_cd_agrees_k5", "correct": 330, "corrections": 18, "regressions": 21, "net_correct": -3}, {"method": "H_A1_g1p0_protect0p025", "correct": 332, "corrections": 14, "regressions": 15, "net_correct": -1}, {"method": "H_A2_g0p25_cd_agrees_k5", "correct": 327, "corrections": 9, "regressions": 15, "net_correct": -6}, {"method": "A2_g1p0", "correct": 332, "corrections": 8, "regressions": 9, "net_correct": -1}, {"method": "Ref2_w0p5", "correct": 335, "corrections": 11, "regressions": 9, "net_correct": 2}, {"method": "C_fixed_low_revert_t0p025", "correct": 329, "corrections": 8, "regressions": 12, "net_correct": -4}, {"method": "A2_g0p25", "correct": 332, "corrections": 1, "regressions": 2, "net_correct": -1}, {"method": "A1_g1p0", "correct": 337, "corrections": 12, "regressions": 8, "net_correct": 4}]

Decision: Retain A1_g1p0(+4), Ref2_w0p5(+2), A2_g0p25(-1) by locked ranking. Hybrid gates fail transfer(-1 to-6); abandon their expansion. Confirmation is now selection data, not blind evidence.

Next action: One frozen official validation stage; no further development iterations.
