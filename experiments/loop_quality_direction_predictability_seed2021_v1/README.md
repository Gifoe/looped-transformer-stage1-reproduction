# Quality direction predictability — 600-question discovery delivery

This delivery implements the user-prioritized **existing600-question analysis** and independently rechecks the completed **1172-question Ouro/LoopCD ARC experiment**. The two experiments use different models and protocols and are never pooled.

The 600-question result is **No-Go for the current selector**: the prespecified linear gate equals Always Euler in nested cross-validation and loses one question in leave-task-out validation. The independent new-data phase of the original prompt is deferred by the user's explicit instruction to deliver1172 and600 first.

| Qwen discovery policy | Original | Always Euler | Linear, nested CV | Linear, leave-task-out | Oracle |
|---|---:|---:|---:|---:|---:|
| Correct /600 |297|314|314|313|329|
| Accuracy |49.50%|52.33%|52.33%|52.17%|54.83%|

All600 IDs are joined against both historical outputs. Of75 prediction changes,32 are corrections,15 degradations and28 both-wrong changes. No gold-derived field enters a selector feature. All old175 sealed artifacts remain unchanged. This analysis uses **zero new GPU hours**.

Read [FINDINGS.md](FINDINGS.md) for Q1–Q10 and limits, [REPRODUCE.md](REPRODUCE.md) to rebuild, and [analysis_plan.json](configs/analysis_plan.json) for fixed features, folds and selection rules.

Artifacts include60 feature columns, all30,600 question/policy/scope predictions, outer and inner splits, fitted parameters and threshold searches,10,000 question-level paired bootstrap comparisons, exact McNemar tests, random controls, per-task results and five PNG/PDF figures. Every policy is scored on all600 questions, including neutral updates. Only the descriptive beneficial/harmful AUC analysis uses the47 nonzero-gain examples.

Historical model: `Qwen/Qwen3-4B-Base` revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`, BF16, block Euler window15–18, K6, alpha1/6. This reuses the historical independent published-recipe implementation; it does not resolve the unavailable author-patch parity limitation.

The separate [1172-question ARC report](../looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026/FINDINGS.md) remains complete. [Fresh raw-score recheck](results/ARC1172_DELIVERY_RECHECK.json) confirms707/722/727 correct. GSM8K was interrupted by a server restart and is not complete; no restart was launched in this delivery.

Directories: `configs/` fixed protocol/rules; `scripts/` CPU pipeline; `data/` gold-free features, separate labels and split inventories; `results/` full policies/statistics/audits; `figures/` five rendered charts; `logs/` provenance and executions; `tests/` leakage and validation contracts; `sources/` primary literature notes. Transport requests and Python caches are excluded from publication.
