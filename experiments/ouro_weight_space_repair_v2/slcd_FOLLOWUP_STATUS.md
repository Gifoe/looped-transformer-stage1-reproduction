# Follow-up target reached: 736/1172 (62.80%)

Frozen R6 `r6_prior_projection` completed ARC-Challenge: **736/1172**, exceeding the user's target of 733/1172 (at least 62.5%). Native K4 / Fixed / Adaptive / incumbent A1 scored 702 / 717 / 716 / 726 under the same fixed-25-shot protocol.

Versus Fixed, R6 rescues 35 questions and damages 16: net +19 (+1.6212 pp), exact McNemar p=0.01097, paired bootstrap 95% CI [0.4266, 2.8157] pp. Versus A1, rescue/damage is 23/13: net +10 (+0.8532 pp), p=0.13250, CI [-0.1706, 1.8771] pp. Incremental superiority to A1 is not established at 0.05.

| Follow-up family | Best train correct/1093 | A1 | Decision |
|---|---:|---:|---|
| R2 Fisher late-residual decorrelation and controls | 689 | 690 | Rejected; no new test |
| R3 candidate-score compositions | 691 | 690 | Only +1; no new test |
| R4 guidance energy / exact KL budgets | 687 | 690 | Rejected; no new test |
| R5 six fixed answer-prior rules | 691 | 690 | Failed +7 training gate; no new test |
| R6 four partial/adaptive prior rules | 705 | 690 | Projection frozen; full test 736 |

R6 selected on reused train 1093: 22 rescues / 7 regressions versus A1. Gains were +6/+1/+8 across the original 384/192/517 subsets and +2/+3/+2/+4/+4 across five fixed reused-training slices. These are descriptive development checks, not independent cross-validation. Matched magnitude scored 692, half strength 698, quarter strength 690. No further method search or test reselection follows the full R6 result.

All scores below are complete-answer summed log likelihood divided once by the same answer character count. Let A=F+(S4-S5), g=F-S4, and e=F_prior-S4_prior; F is token-logit Fixed LoopCD. Prior uses the unedited model with literal `Answer:` prefix. Center g/e across choices, then alpha=clip(dot(center(g),center(e))/dot(center(e),center(e)),0,1), with zero denominator giving 0. Final ranking is A-alpha*e. No gold labels or answer frequencies enter inference. Choice-set dependence is a limitation; this is a calibration/projection adaptation, not a new projection principle or demonstrated causal mechanism.

Final lock SHA256: `6de605b885b4fb9ff72b273ecc0c0484e9fe72fc5d51d793b22367db879b7da7`. Published before test in commit `496602d2416f8114c5227720235c26e5c14e488d`. See [ARC stage report](slcd_R6_ARC_STAGE_REPORT.md), [acceptance audit](results/slcd_R6_TEST_ACCEPTANCE.json), and [full comparison](results/slcd_r6_arc_full_test_results.csv).

Contextual test scores for all 1172 questions were reused after full input/model/protocol compatibility checks; fresh contextual GPU replay coverage in this R6 test is 0. The new auxiliary prior pass completed all 1172 rows. Logical depths are main K5 and prior K4; collection used shared K6 trajectories. No standalone speedup claim is made.

HellaSwag validation 10042 and OpenBookQA test 500 are running with the same frozen R6 rule and all five controls, zero-shot official lm-eval 0.4.9 task configurations. The earlier 1956-row HellaSwag partial file remains unchanged; its scores are source-bound and three saved rows are independently replayed before reuse. Partial external accuracy is not presented as a completed benchmark.

ARC test and all training subsets were exposed in earlier research. This is a reused regression benchmark, not a new blind test. The original A1 lock and older V2 artifacts remain preserved. Research stays in the same folder and original additional six GPU-worker hours ledger; reconnecting did not reset the budget.
