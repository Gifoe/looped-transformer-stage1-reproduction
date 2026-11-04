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

## Restore and verify saved evidence

In this existing experiment directory, fetch every release part listed by the release manifest, then run `python -B scripts/slcd_r6_restore_records.py --verify-only`. The manifest specifies whole archive/part SHA256 and individual record hashes. Omit `--verify-only` to restore absent records; different existing files are refused. Model/env/hidden-state caches and HellaSwag original text are not distributed. HellaSwag numeric scores and exact input hashes are preserved; obtain original data from the pinned source receipts to reconstruct prompts. Full originals remain on the server.

CPU ARC reproduction from restored records: `python -B scripts/slcd_r6_case_audit.py`. This computes only the frozen selected rule, without a model. `python -B scripts/slcd_r6_overlap_sensitivity.py` runs explicitly posthoc sensitivity. External numeric reproduction: `python -B scripts/slcd_r6_public_scores.py verify`, which reconstructs all five methods from exact raw LL and checks paired statistics and every public candidate CSV row. Then rebuild the final report/figures with `python -B scripts/slcd_r6_make_deliverables.py --public-records`. Original full-input server audits require raw inputs not duplicated publicly.

## Original-server inference and full-input audit

Use the unchanged helper/scorer code on this branch and the pinned model/environment in `env` and `models/Ouro-1.4B`. Keep all caches/temp files inside this existing folder. Every GPU job must use `scripts/slcd_supervise.py`, preserving its21600s ledger. `scripts/slcd_r6_prior_worker.py --split test` requires the frozen R6 final lock; `scripts/slcd_r6_final_chain.py` guards transitive hashes before and after evaluating ARC. The prior cache uses exact prefix/continuation/token IDs/model/source keys, never gold or frequencies.

After the complete ARC acceptance gate, external preprocessing uses `scripts/slcd_r6_prepare_external.py` and fixed raw parquets/source receipts. The joint worker is `scripts/slcd_r6_external_worker.py --task hellaswag` (or openbookqa); controller `scripts/slcd_r6_external_chain.py` runs HellaSwag then OpenBookQA. These exact commands are for the original server and require its immutable original main-score authority, raw inputs and caches, including the 1956-row old HellaSwag partial file. The public numeric package supports exact score/statistic reconstruction; it is not by itself the complete original-server runtime snapshot. Source-text-containing authorities remain on the server and are not fabricated or relabeled to bypass their hash locks. Existing completion files validate/reuse immutable evidence; a changed source, model, input, package, or cache raises an error. Inspect failed/stale supervisors before any retry. Re-running on the completed server does not authorize resetting budgets or replacing the final method.

The root `FINAL_METHOD_LOCK.json`, original `slcd_*` R1 test tables/receipts, and `slcd_R1_REPORT_ZH.md` are historical R1 evidence. The current R6 lock is explicitly under configs. Do not rerun method selection or substitute R1 metadata for R6. Original V2 README/REPORT and model weights remain preserved.
