# QKV train-only analysis specification

Recorded UTC: 2026-11-11T17:59:30.095620+00:00

This specification precedes reading any QKV train/validation/test effect. Original execution terms and frozen conditions remain unchanged. This is descriptive analysis after earlier ARC observations, not an independent blind validation.

For each question let s_ij be saved sdpa_R flattened in original head order, a_ij saved post-local-norm a_R, and F_ij the final production score gold-vs-each-nongold gap averaged equally within the question. For each of s, a, and F calculate Q|KV1=21−11, Q|KV2=22−12, KV|Q1=12−11, KV|Q2=22−21, and I=22−21−12+11, per question before aggregation. Report all train1119, non-demo1094, natural four groups and natural no-top-tie.

For vectors calculate each corner norm and the L2 norms and exact-zero counts of each difference and interaction, plus inner product and cosine for each pair of Q differences and each pair of KV differences; zero-vector cosine is undefined. Convert BF16 values to FP64 without modifying saved tensors. Report per-question distributions. Norms in different spaces are not contribution ratios. Report I_s, I_a and I_F separately: final score interaction does not establish internal SDPA interaction.

Project each a corner, differences and I onto the fixed candidate readout direction (W_gold minus average W_wrong) multiplied by fixed final gamma; report fixedwrong in parallel. This is a residual readout coordinate, not the actual causal contribution to the final logit. An optional fixed offline O projection of s must bind O shape and weight hash; the difference between this and a projection also contains local gain and BF16 arithmetic, not only normalization. Optional O must not block raw s and a results.

Final F uses original production scores and parallel exact Fraction computation from saved BF16 target logits. Preserve all candidates, dynamic strongest-wrong gap, all four predictions and rescue/damage. F22−F11 must equal prior L18 G1 per question. Optional equal-background averages phiQ=(Q|KV1+Q|KV2)/2 and phiKV=(KV|Q1+KV|Q2)/2 sum to F22−F11; they divide interaction equally by definition and are not unique or necessary contributions. Retain all sign/zero counts, natural groups, no-tie subsets and counterexamples.

No query-norm control is present: an effect of q2 can reflect direction or magnitude/softmax sharpness. Prior L18 a-norm control does not substitute for a query-norm control. Joint KV swapping cannot separate K addressing from V content. P/Q/R positions do not establish original information source. No query-dominant conclusion is preselected.
