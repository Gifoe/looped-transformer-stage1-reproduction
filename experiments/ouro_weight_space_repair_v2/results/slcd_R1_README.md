# Self-improving training-free LoopCD (original Ouro weights)

Final: **A1_g1p0: 726/1172 vs Fixed717**; net+9, paired95%CI[-0.085,1.621]pp, exactMcNemar p=0.122078.
Actual additional GPU-worker: 0.6129h /6h. This is a previously used regression benchmark, not blind.

Same existing V2 folder; all old files preserved. Entry points:
- [Chinese report](results/slcd_REPORT_ZH.md)
- [Final method lock](FINAL_METHOD_LOCK.json)
- [Iteration log](SELF_ITERATION_LOG.md)
- [Full ARC table](results/slcd_arc_full_test_results.csv)
- [Paired transitions](results/slcd_paired_transitions.csv)
- [Development](results/slcd_development_results.csv), [ablations](results/slcd_ablation_results.csv), [costs](results/slcd_compute_costs.csv)
- [Accuracy](results/slcd_accuracy_comparison.pdf), [rescue/regression](results/slcd_rescue_regression_analysis.pdf)

Model: ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1; original BF16 checkpoint SHA58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. No training or edits.

Reproduction requires the pinned model and the original V1/V2 environment/utility code retained by this branch. Restore compressed records into this existing directory using `python -B scripts/slcd_restore_records.py` (SHA checked; refuses different existing files). Then inspect all locks before inference. Raw state caches remain on server; regenerate them from original model if absent. The original weighted-edit bundles are not inputs to this task.

Development order: iteration1 score rules → original-state token worker → token/orthogonal and failure-feature audit → hybrids → local token neighbors → frozen517 confirmation → frozen top3 validation → one final method lock → full ARC → conditional external tasks only if primary>717. All commands and recipes are in scripts/slcd_*, configs/slcd_* and SELF_ITERATION_LOG.md. Use scripts/slcd_supervise.py for every GPU worker to enforce the independent21600s ledger.

Final scoring entry: `env/bin/python scripts/slcd_finalize.py test`. Token methods first require the locked test worker `scripts/slcd_token_worker_v2.py --split test --job final --recipes configs/slcd_TOKEN_TEST.json`; score-only methods reuse compatible saved baselines. Running against completed immutable receipts validates/reuses outputs. Do not invoke selection again after the final lock or tune against test.

Frozen other-task command (only after ARC gate): `env/bin/python scripts/slcd_external_worker.py --task hellaswag` (and openbookqa if budget). Official harness0.4.9 task config, zero-shot, acc_norm, no retuning. Shared K6 collection gives no standalone runtime speed claim.

External completion or non-completion receipts are under results/slcd_hellaswag* and slcd_openbookqa*; only complete task results are reported.
