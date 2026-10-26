# Research evidence checkpoint — 2026-10-25T19:28Z

This is a partial evidence review, not the final twelve-question answer or a
Go/No-Go decision. It does not create `configs/research_decision.json`. MC output
is complete and independently rebuilt; only Ouro1.4 mathematical validation is
complete. Ouro2.6 validation, both held-out generation tests and actual-exit
timings remain pending. No method, threshold, grid, prompt or test ID was changed
in response to this review.

All MC percentages below use the extension's own disjoint questions and frozen
few-shot/scoring protocol. They must not be pooled with the historical Stage1
rates. Intervals are paired question bootstrap, pointwise and unadjusted.

| Original question | Current evidence and limit | Evidence still required for the final answer |
| --- | --- | --- |
| Q1: training-free positive gains | Qwen A0 58.00%; A3/A4 58.33%, delta +0.33pp with intervals [-1.50,+2.17] and [-1.00,+1.67]pp. These intervals do not establish positive gains or equivalence. A2 38.83%, delta -19.17pp [-23.17,-15.00]. These are independent recipe results; direct author-patch parity remains unverified. | Final source recheck and explicit scope/implementation limitation; no substitution of paper headline numbers. |
| Q2: naive vs damped/anchored | A1 has27 corrections/31 degradations relative to A0; A2 29/144; A3 17/15; A4 9/7. Aggregate gold NLL/token is2.94970/2.97481/3.57541/2.94355/2.94520 for A0..A4. The extension's Qwen likelihood-only files do not measure new hidden-state trajectories. | Interpret any hidden-state evidence only from the separately sealed original experiment; do not infer it from NLL or confidence. |
| Q3: depth monotonicity | Ouro1.4 R1..4:34.00/47.33/49.00/50.00%; Ouro2.6:41.33/54.00/57.00/56.67%. The second sequence is not strictly monotone, but the R3/R4 difference is only one question. Per-question histories include corrections and degradations at both scales. | Separately generated true-depth math results; do not conflate same-prefix MC trajectories with free-running generations. |
| Q4: representative endpoint/reversal paths | R1-correct/R4-wrong counts27/22 and R1-wrong/R4-correct75/68 (1.4/2.6, each N300). R4-anchored CWC counts11/8; WCW17/9. Reversal classes overlap endpoint classes. Cases use first frozen-manifest matches, not manually chosen extreme examples. | Final representative case/trace integrity audit and generation trajectory interpretation. |
| Q5: continuation prediction | At R3, the currently-wrong subsets are N153/129 with15/7 later R4 corrections. Predeclared KL AUC .40966/.39813 and reversed-sign .59034/.60187; hidden-relative-L2 AUC .44348/.36417 and reversed-sign .55652/.63583. Sign reversal shows an association, not no information. Gold-defined conditioning is unavailable to a deployable policy. KL<=.05 subsets have4 corrections/4 degradations among82 and3/3 among115. | Do not infer a causal state-preservation failure from prompt-endpoint KL plus whole-candidate MC answer changes. No causal intervention has run. |
| Q6: LoopCD attribution | Fixed Logits .5 fixes1 CW+8 WW cases and harms9 CC+3 WC cases at1.4; fixes0 CW+5 WW and harms6 CC+1 WC at2.6. Thus some corrected answers are lost as well as some persistently wrong answers fixed. The post-hoc classes use future/gold information. Hidden .5 vs orthogonal control is +2.00pp [.33,4.00] at1.4 but -.67pp [-2.33,1.00] at2.6. | Held-out fixed/adaptive/validation-selected generation results and extraction/cap failures; avoid generalizing one pointwise positive contrast across scales. |
| Q7: compute-matched stopping | At validation target3, none of26 model/policy aggregate results exceeds its own exact-depth-histogram within-benchmark random allocation's central95 upper bound; one is below its lower bound. For KL, test mean depths2.56/2.59, accuracy48.33/56.33%, matched-random means48.19/56.25%. The random central interval is a conditional random-allocation distribution, not an accuracy confidence interval. | Native actual-layer/likelihood parity and randomized repeated walltime measurements; MC simulated saved work alone is not speedup. |
| Q8: prior work overlap | Pinned primary sources and mechanism comparisons are in RELATED_WORK.md; contextual methods are explicitly distinguished from reproduced baselines. Historical search-cache README and independent third-party implementations do not establish author-patch parity. | Reconcile the final observed phenomena with the retrieved primary papers and recheck official source availability before final judgment. |
| Q9: limits on generalization | MC uses three benchmark families, five MMLU subjects, one seed and two related Ouro checkpoints. Scalar-temperature paired contrasts remain necessary controls. Several intervals cross zero. Shared-readout MC costs are not independent method latencies. | Held-out generation, actual stopping and final complete evidence/limitations review. |
| Q10: repeatable failures and next causal test | Corrections/degradations recur at both scales; that alone does not prove hidden-state reusability or irreversibility. No state transplant, hidden freezing or counterfactual continuation intervention was run. | Decide whether failures remain compelling after held-out controls. Any next-stage intervention proposal must identify a causal contrast, prefix identity and matched controls, not treat the observed association as its answer. |
| Q11: next research question | No next-stage direction is selected at this checkpoint. A threshold change or a larger contrast coefficient is not by itself evidence of a new research problem. | Compare completed generation/control/latency results and prior mechanisms before choosing a direction. |
| Q12: final verdict | Pending. Do not infer Go/No-Go from validation gains, small MC deltas, stability labels or an unfinished runtime. | Both validations/tests, actual stopping, source limitation review, full audit, figures and reproducible delivery. |

## Complete Ouro1.4 validation review

These are validation results, not held-out effectiveness estimates. Every
condition has100 questions. Strict and Decimal EM agree in all900 records.
Missing extraction and token-cap counts can overlap; they must not be added as
disjoint error types. No selection was performed by this review.

| Condition | Correct | Missing extraction | Reached512-token cap |
| --- | ---: | ---: | ---: |
| baseline | 88 | 0 | 0 |
| logits_0.25 | 91 | 2 | 1 |
| logits_0.5 | 84 | 2 | 0 |
| logits_1 | 69 | 9 | 7 |
| logits_2 | 17 | 62 | 55 |
| hidden_0.25 | 87 | 1 | 0 |
| hidden_0.5 | 83 | 4 | 1 |
| hidden_1 | 44 | 36 | 36 |
| hidden_2 | 0 | 97 | 96 |

The high-strength negative settings remain in the formal raw data. Audited
generation elapsed times include baseline tracing and the collector's readout
overhead, and are used for the resource budget rather than clean method-speed
comparisons.

Evidence inputs: `results/qwen_metrics.csv`, `ouro_metrics.csv`,
`trajectory_patterns.csv`, `continuation_auc.csv`, `absolute_stability.csv`,
`loopcd_trajectory_attribution.csv`, `readout_pairwise_controls.csv`,
`halting_selected.csv`, `halting_matched_random.csv`,
`math_validation_pretest_review_Ouro-1.4B.json`, and their original JSONL.
The complete1.4 validation raw SHA256 is
`50de00048ce2c68d234551682cf0b1df48ac17915d6f3ab8bd6cd119d7ae7bf4`.
Its complete-prefix CPU audit checked900 records and13,575 baseline token
positions independently of the server, without model weights.
