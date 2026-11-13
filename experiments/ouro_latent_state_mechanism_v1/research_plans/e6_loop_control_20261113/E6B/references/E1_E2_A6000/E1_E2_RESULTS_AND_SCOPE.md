# E1 and E2: completed results and scope

## E1: SDPA interaction and the downstream suffix

For each query background let Fij be the original production gold-versus-mean-non-gold candidate gap and let Fadd be the score after injecting BF16(FP64(S21)+FP64(S12)−FP64(S11)) at the L18 SDPA R output into the fixed K1 receiver. The exact scalar identity is I_terminal = (F22−Fadd) + (Fadd−F21−F12+F11). The first term measures the effect of replacing the actual S22 by this additive intervention, including its final BF16 cast; the second is the non-additivity of the same downstream suffix. The latter contains o_proj rounding, local normalization, residual/MLP and later layers and is not attributable to the MLP alone. The effective removed vector is not a pure exact-real attention interaction: finite-precision and additive cast residual are included. Per-candidate scores, fixed/dynamic alternatives, sign-sensitive Fraction results and 13 stage diagnostics are retained in the linked machine-readable files.

| Split and natural group | N | Q1 removed | Q1 suffix | Q1 total | Q2 removed | Q2 suffix | Q2 total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train remainder non-demo rescue |336|0.418744|−0.060516|0.358228|0.512463|−0.051060|0.461403|
| Validation rescue |103|0.388046|−0.052083|0.335963|0.506776|−0.056331|0.450445|
| Test rescue |378|0.393794|−0.054481|0.339313|0.499311|−0.048060|0.451251|
| Train remainder non-demo damage |31|0.000672|0.038306|0.038978|−0.020833|0.011425|−0.009409|
| Validation damage |9|−0.239583|0.030093|−0.209491|−0.177083|0.023148|−0.153935|
| Test damage |51|−0.009804|0.040033|0.030229|−0.057190|0.022876|−0.034314|

Thus, in natural rescue samples, the measured component removal has a positive average effect larger than the resulting final interaction because the suffix term averages negative. This is a conditional intervention, not proof that the same component is necessary on the natural K2 path. Train damage Q1 violates the pre-frozen expectation of a strictly negative mean (+0.000672); validation and test do not erase the failure. Validation damage N=9 is especially small. Rescue removal exceeds the unrelated control on average: train Q1/Q2 0.418744/0.512463 versus 0.289373/0.371435, validation 0.388046/0.506776 versus 0.220570/0.358313, test 0.393794/0.499311 versus 0.265597/0.353230. Matched SDPA-output norm behaves similarly, which is not strong evidence of a uniquely directional mechanism because downstream bias-free projection and local RMS normalization are approximately scale invariant.

The 64-case discovery was stratified into four groups of 16 and is not a population estimate. The held-out train remainder has 1030 non-demo questions; 25 train demonstrations are reported separately. Validation and Test are within-project comparisons, not previously unseen ARC experiments. Original natural tensors deleted earlier were not silently reread: restored original metadata plus audited surviving natural-branch tensors were used. Train/validation/test saved-score independent aggregation is not an independent model forward. Three full-model FP32 cases contain no natural rescue; FP32 used actual efficient SDPA while the BF16 kernel was not profiled, so their comparison cannot isolate dtype or establish rescue robustness.

## E2: fixed probes and state interventions

A masked linear probe was fitted on Train non-demo, with validation-selected regularization and frozen once before Test. Test was previously exposed in the larger project; a prior test candidate OOV audit and partial validation exposure in aborted fitting attempts are recorded. All bias parameters are L2-regularized in the final fit after a candidate class with no positive label made the earlier unregularized-bias objective lack a finite optimum. Unit-L2 inputs were subsequently standardized per feature and therefore are not a pure angular control. Shuffling leaves two singleton candidate sets unchanged.

For the pre-specified raw L18-input probe trained on K1, Test accuracy was 0.4224 on K1 states and 0.6664 on K2 states; its permuted-label counterpart was 0.2577 and 0.2287. Native K1/K2 accuracies were 0.5341/0.8131. Learned-probe decodability does not establish the native model's causal use of the decoded information. Candidate-conditional probabilities are renormalized diagnostics; the production model scores remain unchanged. Full two-site/two-loop/raw-and-unit/control tables, groups, per-question predictions and counterexamples are in E2_FROZEN_TEST_REPORT.md and its associated machine-readable results.

In the stratified 64-case E2.3 state intervention, rescue-group rollback of the fixed train-derived probe subspace changed the candidate gap by −0.3724, versus −0.2083 for an unrelated control, and damaged 2/16 decisions versus 0/16; the orthogonal-complement rollback changed the gap by −1.7135 and damaged 4/16. This supports an effect in the tested intervention, not a claim that the learned subspace is the only or dominant natural causal path. The 64 groups are balanced by design and not population proportions; review E2.4 and R producer reports separately.

The audited state extraction covers the saved R position across 24 physical layers and both loops for x/a/y/b/z, plus the recorded loop boundaries; it does not represent the full prompt state. The fixed global rank-4 probe span includes an extra E-axis that was not an offered answer in most questions; that axis accounts for a median 95.77% of squared projected change energy. The offered-candidate rank-3 span is a pre-specified sensitivity analysis, not an established answer subspace. The perturbation sizes are different: across the balanced 64, the mean norm is approximately 4.084 for the complement versus 1.394 for the global P span and 0.320 for the offered span. The larger gap change under complement/full rollback therefore cannot establish that its information is intrinsically more important than P.

E2.4 separately crossed the actual residual and attention increment: C00=(x2,a2), C11=(x1,a′), C10=(x1,a2), C01=(x2,a′), where a′ is the actual local attention output after the R input x1 is placed in the K2 context. In the 16 natural-rescue cases the respective correct counts are 16/11/14/16. These are controlled mixed-state compatibility observations, not evidence that an attention component mediates a fixed percentage of the natural recovery; C10's three additional correct cases relative to C11 must not be relabeled as a 60% causal restoration.

## Evidence index and outstanding work

- `source/E1/remainder_analysis`, `validation_analysis`, `test_analysis`: full audited compact rows, summaries, Fraction sensitivity and independent saved-score aggregation.
- `source/E1/*cuda_audit.json`: independent CUDA saved-logit/source audit ledgers; Test SHA f17c4c29086708211f4769c6b70962d74eac3cc21b4b78d6b35a204d9489fc64.
- `source/E2/probe_test_once/`: frozen Test analysis and reproducibility code; `local_reviews/handoff/E2_V3_INDEPENDENT_AUDIT.md` is the independent implementation audit.
- `local_reviews/state_intervention/`: 64-case E2.3/E2.4 and R producer evidence and limitations.
- `local_reviews/fp32_sensitivity/results_snapshot/`: the three-case full-model FP32 report and saved-output audit.
- `source/E3/`: completed discovery and producer evidence; ongoing 4090 A-only Train/Validation writer holdout is PENDING, not a finding in this snapshot.

This report is a Terra synthesis of existing results, not a newly signed Astra review. Historical Astra decisions are retained with their original provenance. Model weights, private credentials and large raw tensors are excluded. Reproduction requires the separately retained checkpoint, original input/source ledgers, and the scientific source paths and dependencies recorded in the receipts; compact summaries alone do not reproduce the GPU model forward.
