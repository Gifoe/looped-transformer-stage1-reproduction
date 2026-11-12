# E4 state-to-attention controlled mediation

Final status: K2/K3/K4 full 299 and preregistered128 confirmation are complete. ConfirmationK2 N/U/restored-U/A-only=103/98/105/101;9 natural corrections lost,6 restored,2 strict all-control cases. The direct meanwrong effect−1.353923 remains large; restoration+0.040446 has interval[-0.0160,+0.0970]. Late restoration means are negative in both cohorts and late strict confirmation recoveries are 0. See FINAL_RESEARCH_REPORT.md for closed-cohort synthesis.

Current result: the complete 299-question Validation intervention supports an L18 attention-mediated decision path for a subset of natural K2 corrections, alongside substantial continuous effects that bypass this specific attention mediator. It does not establish where the earlier recurrent computation generated the useful state. All four corners and controls pass independent source, score and tensor audits.

At natural K2 L18/R, X2 is the current residual and X1 the same-question native K1 residual. U changes only this last-token input and computes its actual A(U) with K2 non-R states. U_restore retains X1 and actual U QKV but restores native K2 SDPA_R A2. A_only retains X2 and injects A(U). Actual A(U) differs bitwise from natural K1 A1 on all 299 questions. The restored normalized attention increment exactly matches native K2, while non-R x/Q/K/V/SDPA hashes remain unchanged.

| Condition | Correct / 299 | Rescue / damage vs N | Meanwrong Δ | Fixedwrong Δ | Dynamic margin Δ |
|---|---:|---:|---:|---:|---:|
| identity | 243 | 0 / 0 | +0.000000 | +0.000000 | +0.000000 |
| U | 220 | 9 / 32 | -1.453247 | -1.107651 | -1.248328 |
| U_restore | 240 | 6 / 9 | -1.375505 | -1.042433 | -1.119983 |
| A_only | 235 | 4 / 12 | -0.130121 | -0.108905 | -0.148620 |
| state_norm_only | 243 | 1 / 1 | -0.096572 | -0.069398 | -0.070234 |
| state_unrelated_delta | 228 | 6 / 21 | -0.452881 | -0.249164 | -0.455477 |
| mediator_norm_only | 223 | 9 / 29 | -1.451644 | -1.102007 | -1.243102 |
| mediator_unrelated_delta | 216 | 6 / 33 | -1.467618 | -1.149457 | -1.301003 |

N has 243 correct. State controls recompute attention normally. Mediator controls retain the same X1 block, then substitute an endpoint-norm-only or unrelated matched-delta attention vector. They use separate ID-sorted donor maps frozen before outcomes. All effects above use the unchanged N baseline; mediator recovery effects below use U, avoiding baseline confusion.

Among the 104 same-hardware natural K1→K2 corrections, U loses 29 and A2 restores 22 of those 29. The norm and unrelated mediator controls restore 3 and 6 respectively. Fifteen restored cases exclude native/damaged/restored top ties and are not restored by either mediator control; eleven additionally remain correct under both input-state controls. These are nested specificity criteria, not two conflicting counts. The frozen intervention was run on all questions; this case classification is post hoc descriptive accounting. Twenty-two out of 29 is not a population mediated fraction.

U_restore is correct on 96/104 natural corrections, compared with 75 under U and 104 under N. Seven U-damaged corrections remain wrong after restoration; one additional correction becomes wrong only under restoration. Across all 299, U damages 32 and restoration recovers 25, but creates two new damages, yielding9 total damages relative to N. A_only loses 11 natural corrections. The conditional state and attention effects are not interchangeable with isolated whole-QKV rollback.

| Contrast | Meanwrong | Fixedwrong | Dynamic margin | Global-K1-fixedwrong |
|---|---:|---:|---:|---:|
| total | -1.453247 | -1.107651 | -1.248328 | -1.530518 |
| attention_only | -0.130121 | -0.108905 | -0.148620 | -0.183319 |
| direct_at_native_attention | -1.375505 | -1.042433 | -1.119983 | -1.415134 |
| restore_effect | +0.077742 | +0.065217 | +0.128344 | +0.115385 |
| interaction | +0.052379 | +0.043687 | +0.020276 | +0.067935 |
| norm_restore_effect | +0.001603 | +0.005644 | +0.005226 | +0.004181 |
| unrelated_restore_effect | -0.014371 | -0.041806 | -0.052676 | +0.004181 |

A large negative continuous effect persists when A2 is fixed: U_restore−N has meanwrong −1.375505 and is negative on 258/299 questions (38 positive,3 zero). This supports a controlled path through the altered residual and subsequent computation bypassing this L18 attention mediator. It does not identify a pure MLP route: later attention, normalization and residual computation remain intact. The substantial discrete restoration is also real; mean-gap dominance cannot erase threshold crossings. These statements describe different outcomes and are compatible.

Numerical checks: pilot 8 FP32-SDPA/BF16-writeback U diagnostics change no predictions. This is not full-model FP32. Two Validation mediator-unrelated vectors differ from independent Windows CPU FP32 reconstruction by one BF16 element each (3.8147e−6 and 7.4506e−9); FP64 rounding-bound checks pass and same-producer-environment CPU recomputation is bitwise identical. All original raw evidence is retained. See SAME_ENV_CONTROL_RECOMPUTATION.json and control_math_cross_cpu_rounding in the independent audit.

Counterexamples include ACTAAP_2007_7_25, where U and restored-U remain wrong despite both input-state controls retaining correctness; NCEOGA_2013_8_4, where U remains correct but restoring A2 creates a new error; and the two pilot losses, neither uniquely rescued against both mediator controls. The pilot and full Validation are reported separately. Mixed states can have compatibility effects, and one unrelated direction cannot exclude every generic perturbation explanation.

Evidence: results/ASTRA_MEDIATION_VALIDATION299_REVIEW_INDEPENDENT.json preserves all groups, four metric definitions, exact raw-logit signs and complete damage cases; ASTRA_MEDIATION_DECISION_CASES.json preserves every question and both strict criteria. Raw tensors and candidate logits are in results/mediation_validation299_v1. Complete Validation collector time was 487.02s (494.49s UTC command window),2691 recorded forward windows. Resource peaks apply only to recorded forward windows, not end-to-end loading/derivation/saving.

Order 005 and 006 are complete; late-depth outcomes and confirmation limitations are reported in the final synthesis.

Post hoc common no-tie robustness:271 questions have no adjacent-native or four-corner top tie. On this fixed common subset, U / U_restore / A_only meanwrong effects are -1.536132, -1.464041, -0.126422. No stored condition changes its predicted candidate under the auxiliary BF16-logsoftmax scoring diagnostic. This does not replace the separate FP32-SDPA check. Full subset accounting is in ASTRA_MEDIATION_K2_TIE_SENSITIVITY.json.
