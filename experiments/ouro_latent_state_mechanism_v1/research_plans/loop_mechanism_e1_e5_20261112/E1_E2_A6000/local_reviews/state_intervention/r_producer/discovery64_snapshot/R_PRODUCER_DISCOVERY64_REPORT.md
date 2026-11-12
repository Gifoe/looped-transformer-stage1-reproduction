# R producer coarse cuts: frozen discovery64

Three cut interventions insert same-question natural K1 R state at the input of L1/L9/L17 in a natural K2 receiver; direct L18 R rollback is an already-audited target anchor. R is the final token; non-R L18 inputs were bitwise checked and the independently implemented CUDA replay rebuilt full inputs and matched hashes, a_R, full logits and scores. Full sequences were not archived in the 64 result. Four natural groups were balanced at 16 each; the combined sample is not a population estimate.

L_cut=G_nativeK2−G_cut and L_target−L_cut=G_cut−G_L18anchor. Positive target-minus-cut indicates smaller terminal gap loss than direct target rollback; it is not a recovery percentage, cut importance or evidence of a writer. Intervention norms and geometry differ across cuts and are not matched controls.

| Group / upstream input | N | L_cut gold-allwrong mean / median | L_target−L_cut mean / median | Wrong / rescued | Mean injected Δ L2 | Mean L18 Δx L2 | Mean L18 Δa L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all / L1 | 64 | 1.6076 / 1.1198 | -0.5158 / -0.4792 | 9 / 9 | 31.3460 | 8.8689 | 1.3595 |
| all / L9 | 64 | 0.5309 / 0.4375 | 0.5609 / 0.3438 | 1 / 4 | 3.8587 | 3.9396 | 0.6667 |
| all / L17 | 64 | 1.0641 / 0.7917 | 0.0277 / 0.0208 | 5 / 6 | 4.4215 | 4.2757 | 0.8578 |
| rescue / L1 | 16 | 2.3945 / 2.4062 | -0.3424 / -0.1510 | 5 / 0 | 30.4907 | 8.8431 | 1.4160 |
| rescue / L9 | 16 | 1.0378 / 1.0208 | 1.0143 / 0.8854 | 1 / 0 | 3.8972 | 3.9290 | 0.6447 |
| rescue / L17 | 16 | 1.9987 / 1.6667 | 0.0534 / 0.0208 | 5 / 0 | 4.6232 | 4.4525 | 0.9773 |
| damage / L1 | 16 | 0.2458 / 0.3333 | -0.4463 / -0.4948 | 0 / 1 | 30.4965 | 8.8367 | 1.3986 |
| damage / L9 | 16 | 0.0807 / 0.0208 | -0.2812 / -0.1771 | 0 / 2 | 3.8007 | 3.9034 | 0.7394 |
| damage / L17 | 16 | -0.1797 / -0.0625 | -0.0208 / -0.0104 | 0 / 5 | 4.3562 | 4.2276 | 0.8750 |
| stable_correct / L1 | 16 | 3.7178 / 3.8490 | -1.5081 / -1.2318 | 4 / 0 | 33.1635 | 8.8733 | 1.2804 |
| stable_correct / L9 | 16 | 0.9740 / 0.7396 | 1.2357 / 1.3750 | 0 / 0 | 3.8844 | 3.8954 | 0.5601 |
| stable_correct / L17 | 16 | 2.1562 / 2.1250 | 0.0534 / 0.0521 | 0 / 0 | 4.3691 | 4.2365 | 0.7570 |
| stable_wrong / L1 | 16 | 0.0723 / -0.1094 | 0.2337 / 0.4271 | 0 / 8 | 31.2332 | 8.9225 | 1.3430 |
| stable_wrong / L9 | 16 | 0.0312 / -0.0521 | 0.2747 / 0.1667 | 0 / 2 | 3.8525 | 4.0305 | 0.7226 |
| stable_wrong / L17 | 16 | 0.2812 / 0.3021 | 0.0247 / 0.0208 | 0 / 1 | 4.3377 | 4.1864 | 0.8218 |
| natural_no_top_tie / L1 | 57 | 1.7834 / 1.2969 | -0.6332 / -0.6146 | 9 / 6 | 31.6786 | 8.9085 | 1.3753 |
| natural_no_top_tie / L9 | 57 | 0.5654 / 0.5000 | 0.5848 / 0.3750 | 0 / 2 | 3.8617 | 3.9229 | 0.6567 |
| natural_no_top_tie / L17 | 57 | 1.1166 / 0.8333 | 0.0336 / 0.0208 | 4 / 5 | 4.4088 | 4.2627 | 0.8487 |

All three gap definitions, means, medians, float and exact raw-BF16 Fraction signs with per-question IDs, gold/fixed/dynamic contrasts, raw scores, x_R relative distance/cosine, actual a_R distances, diagnostic frozen global/offered projections and fixed-readout lens changes are preserved in summary.json and per_question.jsonl. Fixed-readout lens is an offline diagnostic, not a natural intermediate prediction. A small terminal loss after an early cut does not imply hidden-state reconstruction; inspect actual x/a differences.

Independent CPU source audit checked same-question branch donor and L18 target x/a/logits, SHA ce4f73ec5763d1852d4f38fb6079be6353777193623459ee5eb75561bb94242d. Independent bare-hook CUDA model replay checked all 192 cuts and full-logit/a/score and full input hashes, SHA c059cc05958ad5195d41671e9d5f682e60621c49620d6032929c306b4f24ca7d. Independent saved-score recomputation passed. These tests localize effects of substitutions; they do not establish the upstream computation, encode a natural producer or supply an independently held-out validation.
