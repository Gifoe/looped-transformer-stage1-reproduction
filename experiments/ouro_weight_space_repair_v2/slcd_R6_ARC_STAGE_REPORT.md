# Frozen R6 ARC result:736/1172 (62.80%)

The full ARC-Challenge target was reached: **736/1172 (62.7986%)**, versus the required733. HellaSwag10042 and OpenBookQA500 are now being evaluated with this same frozen method.

| Method | Correct/1172 | Accuracy |
|---|---:|---:|
| native4 | 702 | 59.8976% |
| fixed | 717 | 61.1775% |
| adaptive | 716 | 61.0922% |
| incumbent_A1 | 726 | 61.9454% |
| r6_prior_projection | 736 | 62.7986% |

Versus Fixed: 35 rescues, 16 regressions, net+19 (1.6212pp). Paired50k-bootstrap95%CI[0.4266,2.8157]pp; exactMcNemar p=0.0109736.

Versus incumbent A1: 23 rescues, 13 regressions, net+10 (0.8532pp). CI[-0.1706,1.8771]pp; p=0.132498. The incremental A1 comparison is not statistically supported at0.05.

## Frozen scoring rule

All S values are complete-answer log-likelihood sums divided by the same answer character count. F is token-logit Fixed LoopCD, not a linear combination of separately normalized loop scores.

Let A=F+(S4-S5), g=F-S4, and e=F_prior-S4_prior. The prior uses the same original model with fixed Answer: prefix. Center g/e across the current answer choices, then alpha=clip(dot(center(g),center(e))/dot(center(e),center(e)),0,1), with zero denominator giving0. Final ranking score is A-alpha*e.

The projection coefficient uses no gold labels. It depends on the answer-choice set; adding/removing choices can alter alpha. Projection/calibration are existing tools; this is a tested recurrent-guidance adaptation, not a claim of a new projection principle.

## Protocol and integrity

Original ByteDance/Ouro-1.4B revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1, unedited BF16 weights, GB10; same global fixed25 ARC demonstrations/seed2021/lm-eval0.4.9/acc_norm. All1172 saved contextual scores were compatibility checked and reused; new original-context GPU replay coverage in this R6 test is0. Full auxiliary prior collection1172 rows completed; no independent historical prior baseline is claimed.

Final lock SHA256: `6de605b885b4fb9ff72b273ecc0c0484e9fe72fc5d51d793b22367db879b7da7`. It was published before this test in commit496602d2416f8114c5227720235c26e5c14e488d. Only R6 projection was tested after the follow-up training search; R2–R5 were not evaluated on new test variants.

The same ARC test was used in earlier research, including exposed A1=726. It is a reused regression benchmark, not a new blind test. Training1093 and its fixed slices were also reused during development. Paired tests above describe this benchmark and do not establish broad unseen-task superiority.

Logical depths are mainK5/priorK4, with main vocabulary-head depths1/4/5 and prior1/4. Collection actually used sharedK6 trajectories; no standalone latency or speedup claim is made.

[Final lock](configs/slcd_R6_FINAL_LOCK.json) · [Full table](results/slcd_r6_arc_full_test_results.csv) · [Paired transitions](results/slcd_r6_paired_transitions.csv) · [Acceptance audit](results/slcd_R6_TEST_ACCEPTANCE.json)
