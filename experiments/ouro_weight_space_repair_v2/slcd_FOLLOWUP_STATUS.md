# Frozen R6 training-free LoopCD: 736/1172 (62.80%)

Original BF16 Ouro-1.4B, fixed25-shot ARC, no training or weight editing. Target733 reached. Against Fixed717: rescue35/damage16, net+19, +1.6212pp, exactMcNemar p=.01097. Against A1=726:23/13, net+10; p=.13250, not reliable at.05. ARC is a historically reused regression benchmark.

| benchmark | method | correct | n | accuracy_percent | net_vs_fixed |
| --- | --- | --- | --- | --- | --- |
| ARC-Challenge | native4 | 702 | 1172 | 59.8976 | -15 |
| ARC-Challenge | fixed | 717 | 1172 | 61.1775 | 0 |
| ARC-Challenge | adaptive | 716 | 1172 | 61.0922 | -1 |
| ARC-Challenge | incumbent_A1 | 726 | 1172 | 61.9454 | 9 |
| ARC-Challenge | r6_prior_projection | 736 | 1172 | 62.7986 | 19 |
| HellaSwag | native4 | 7203 | 10042 | 71.7287 | -28 |
| HellaSwag | fixed | 7231 | 10042 | 72.0076 | 0 |
| HellaSwag | adaptive | 7235 | 10042 | 72.0474 | 4 |
| HellaSwag | incumbent_A1 | 7239 | 10042 | 72.0872 | 8 |
| HellaSwag | r6_prior_projection | 7259 | 10042 | 72.2864 | 28 |
| OpenBookQA | native4 | 206 | 500 | 41.2000 | -5 |
| OpenBookQA | fixed | 211 | 500 | 42.2000 | 0 |
| OpenBookQA | adaptive | 211 | 500 | 42.2000 | 0 |
| OpenBookQA | incumbent_A1 | 213 | 500 | 42.6000 | 2 |
| OpenBookQA | r6_prior_projection | 219 | 500 | 43.8000 | 8 |

All external tasks use official0-shot prompts and the exact frozen rule. No tuning after test.

- [Chinese report, nine questions](results/slcd_REPORT_ZH.md)
- [Current R6 final lock](configs/slcd_R6_FINAL_LOCK.json), [external lock](configs/slcd_R6_EXTERNAL_LOCK.json)
- [Frozen score implementation](scripts/slcd_r6_scores.py)
- [Full ARC table](results/slcd_r6_arc_full_test_results.csv), [candidate scores](results/slcd_r6_arc_candidate_scores.csv), [paired transitions](results/slcd_r6_paired_transitions.csv), [cases](results/slcd_R6_FAILURE_CASES.md)
- [HellaSwag full table](results/slcd_r6_hellaswag_results.csv), [OpenBookQA full table](results/slcd_r6_openbookqa_results.csv)
- [Development](results/slcd_R6_all_followup_development.csv), [initial ablations](results/slcd_ablation_results.csv), [all iterations](SELF_ITERATION_LOG.md), [costs](results/slcd_R6_compute_costs.csv)
- [Accuracy figure](results/slcd_accuracy_comparison.pdf), [paired analysis](results/slcd_rescue_regression_analysis.pdf)
- [ARC independent audit](astra/slcd_R6_FINAL_CLAIM_AUDIT.md), [external acceptance](results/slcd_R6_EXTERNAL_ACCEPTANCE.json), [overlap sensitivity](results/slcd_R6_OVERLAP_SENSITIVITY.json)
- [Release manifest](artifacts/slcd_R6_RELEASE_INDEX.json) and [record contents](artifacts/slcd_R6_RECORDS_CONTENTS.json)

Model revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1; safetensors SHA58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. Environment and input hashes are locked. Logical mainK5/priorK4, actual sharedK6 collection; no latency claim. Additional worker total 3.7681h/6h, excluding historical cache generation.


Final deliverables complete; see the current Chinese report and release manifest.
