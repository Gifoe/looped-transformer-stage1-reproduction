# Ouro R7 prefix-conditioned prior projection

Primary **M2_N0_token**, complete ARC-Challenge **740/1172**; R6 **736/1172**, net **+4**. Target751 not reached. 用户随后要求停止另外两个数据集并继续ARC方法研究：已完成的OpenBookQA负结果保留，HellaSwag部分采集已停止，不报告其完整准确率。R7公式及ARC结果保持冻结。

All work remains in the original V2 directory; new files user7_ names. Read [Chinese report](results/r7_REPORT_ZH.md), [locked method](configs/r7_FINAL_METHOD_LOCK.json), [test metrics](results/r7_arc_full_test_results.csv), [paired transitions](results/r7_paired_transitions.csv), [compute accounting](results/r7_compute_costs.csv).

Replay completed numerical records onCPU: `python -B scripts/r7_evaluate.py --stage arc`. Rebuilding scores requires the original pinned model, fixed manifests and environment from the base repository. Inference must use the original no-KV/no-mask BF16 path; the historical KV optimization failed parity. `r7_dispatch.py` refuses duplicate failed attempts and never selects an alternate test winner. `r7_deploy.py` separately profiles exact5/4 deployment against development scores.

The branch also carries the unchanged inheritedR6 multipart release. Restore its inputs with `python -B scripts/slcd_r6_restore_records.py`, then verify/restore new numerical evidence with `python -B scripts/r7_restore.py`. `--verify-only` checks every indexed byte without overwriting. Historical raw states are not required by `scripts/r7_replay_fresh.py`, which reruns the frozen formula from the original model and manifests and compares every scored vector to archived results.

Inherited ARC question/rendered-demo data retain [CC BY-SA4.0 attribution](data/DATA_SOURCE_NOTICE.txt). Model weights, environment, credentials, transport and raw historical hidden-state tensors are excluded from the public evidence archive; historical states remain regenerable under their SHA authorities. Dataset histories were already exposed; no blind-test claim.
