# Frozen R6 LoopCD: ARC 736/1172 (62.80%)

The full ARC target of at least62.5% is reached. Fixed717, Adaptive716, NativeK4 702, A1 726; R6 prior projection736. No training or weight editing.

Against Fixed:35 rescues/16 damage, net+19 (+1.6212pp), exactMcNemar p=.01097, paired95%CI[.4266,2.8157]pp. Against A1:23/13, net+10; p=.13250 and CI crosses0, so reliable incremental superiority is not established.

- [ARC stage report](slcd_R6_ARC_STAGE_REPORT.md)
- [Chinese report (external tasks pending)](results/slcd_REPORT_ZH.md)
- [Current R6 frozen lock](configs/slcd_R6_FINAL_LOCK.json), [external lock](configs/slcd_R6_EXTERNAL_LOCK.json)
- [Frozen ranker](scripts/slcd_r6_scores.py), [full ARC results](results/slcd_r6_arc_full_test_results.csv), [paired transitions](results/slcd_r6_paired_transitions.csv)
- [Independent ARC audit](astra/slcd_R6_FINAL_CLAIM_AUDIT.md), [all changed cases](results/slcd_R6_FAILURE_CASES.md), [candidate vectors](results/slcd_r6_arc_candidate_scores.csv)
- [Posthoc split-overlap sensitivity](results/slcd_R6_OVERLAP_SENSITIVITY.json)
- [Accuracy figure (ARC only at this stage)](results/slcd_accuracy_comparison.pdf), [paired figure](results/slcd_rescue_regression_analysis.pdf)
- [All ten iterations](SELF_ITERATION_LOG.md), [follow-up status](slcd_FOLLOWUP_STATUS.md)

HellaSwag validation10042 and OpenBookQA test500 are running with the same frozen method, official lm-eval0.4.9 zero-shot prompts and five controls. No partial external accuracy is reported. The server's CPU finalizer waits for the existing chain, verifies full inputs/caches/statistics, creates lossless numeric records, and builds final reports/plots. Publication of those final artifacts is still pending.

ARC is a historically exposed regression benchmark. Contextual scores were reused after full compatibility checks, fresh contextual GPU replay coverage0; new auxiliary prior pass covers1172 rows. Logical mainK5/priorK4, actual sharedK6; no standalone speed claim. The root FINAL_METHOD_LOCK.json and original slcd ARC receipts/tables are historical R1, not R6. Old V2 README/REPORT and weights remain unchanged.

Original model ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1; BF16 GB10, fixed25 ARC demos, seed2021, acc_norm with one character-length denominator. CPU ARC verification: `python -B scripts/slcd_r6_case_audit.py` after obtaining raw saved ARC scores/manifest/prior files from the release or server. No test reselection is authorized.
