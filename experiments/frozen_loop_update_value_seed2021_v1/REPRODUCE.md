# Reproduction

This experiment starts from public branch `arc1172-qwen600-quality`, commit `06e40a3404f5f2f01418a52bad6942968a8c5fdf`. Its own pre-outcome protocol/code commit is `967d2eb` on `update-value-ouro14-20261029`. Earlier sealed experiments remain separate.

## Directories on the execution server

* Experiment: `D:/Codex_Research/experiments/frozen_loop_update_value_seed2021_v1`.
* Existing pinned checkpoint: `D:/Codex_Research/experiments/looped_stage1_trajectory_audit_seed2021_v1/models/Ouro-1.4B`.
* Python: `D:/Codex_Research/environments/loopbase_env_20261025/Scripts/python.exe`.
* Inherited dependencies: sibling historical project's `.venv/Lib/site-packages`.
* Historical accounting: sibling historical project's `extensions/baseline_suite_20261025_seed2021/logs/gpu_budget.json`.
* Local artifact mirror: `D:/chenyu-iclr/experiments/frozen_loop_update_value_seed2021_v1`.

The server's old D-root directories were migrated before this run. `runtime_bootstrap.py` repairs the stale inherited package path within this process only, preserving the original precedence (experiment environment, inherited project packages, system packages). It does not upgrade or edit shared environments. Windows affinity code applies the previous verified `0xffff0000` mask to this process only; this mask is specific to this server. Linux skips it. Do not copy that hardware mask blindly to a different Windows CPU.

## Resources and versions

Use official `ByteDance/Ouro-1.4B` revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. The checkpoint metadata must match before loading. The historical resource acquisition scripts and manifests document the original download. No weights are published in this result repository. Keep the old data manifests available for the exposure audit. Dataset revisions and parquet hashes are recorded in `data/manifest_audit.json`; harness templates are retained in `sources/`.

The execution environment uses Python 3.13.9. Pin lm-evaluation-harness 0.4.9 and Transformers 4.57.1. The executed runtime and model loading diagnostics are in `results/model_load.json`. Use a CUDA-enabled PyTorch supporting the RTX5090. Frozen backbone, BF16, SDPA, no quantization, no gradient training of the language model.

## Execution order

Run from the experiment directory using the stated Python:

```text
python -X utf8 scripts/prepare.py
python -X utf8 scripts/supervise.py pilot
python -X utf8 scripts/supervise.py discovery
python -X utf8 scripts/analyze.py discovery
python -X utf8 scripts/supervise.py head_train
python -X utf8 scripts/supervise.py validation
python -X utf8 scripts/analyze.py train
python -X utf8 scripts/supervise.py final_sciq
python -X utf8 scripts/supervise.py final_piqa
python -X utf8 scripts/finalize_analysis.py
python -X utf8 scripts/audit_results.py
python -X utf8 scripts/report_worker.py diagnose_signals.py
python -X utf8 scripts/report_worker.py render_report.py
python -X utf8 scripts/report_worker.py render_appendix.py
python -X utf8 scripts/refine_findings.py
```

Inspect pilot parity/cost before discovery and the discovery report before optional head collection/training. Do not run final stages until `configs/FINAL_LOCK.json` exists. Final analysis verifies its locked source hashes. The final manifests and predictions are immutable; scripts refuse to overwrite a prior prediction file. Reproduce in a fresh directory or preserve old outputs and document an explicit new run. Reads use physical JSONL lines, including questions containing Unicode line separators.

For durable execution over Windows SSH, use `scripts/launch.ps1 <stage>` via Win32_Process.Create; a plain SSH-child Start-Process did not survive the session on this host. The launcher and supervisor retain native exit records. A missing process or observation timeout alone is not evidence of completed inference. All failed worker wall time is charged conservatively, including model import/loading; no historical budget is reset.

## Audits and interpretation

`tests/test_features.py` verifies gold/ID invariance, answer-option permutation invariance, and paired correction arithmetic. The GPU pilot checks native final readout equality, official HFLM likelihood/prediction parity, causal question-token probes and matched Adaptive scoring. Float32 log-softmax versus BF16 harness calculations have a declared per-token tolerance0.05; prediction parity is required.

The primary metric is character-normalized full-candidate likelihood accuracy; raw accuracy is also reported. SciQ includes official supplied evidence. PIQA contains 1802 eligible validation questions after 36 normalized duplicate question exclusions, so it is not the unfiltered 1838-question score. Exposure checking cannot rule out pretraining contamination.

Known-question predictive loss is a hypothesis, not a verifier. Format changes add no independent facts. Oracle choices are unreachable without labels. Retrospective rollback after all four loops saves no backbone computation. A No-Go result applies to the tested signals, supervision and action set, not all possible frozen-model controllers.

## Locks, integrity, and CPU-only inspection

`configs/PREVALIDATION_LOCK.json` hashes the inference/analysis code, protocol and all five data manifests before validation. `configs/FINAL_LOCK.json` then freezes fitted heads, selected policy and validation-selected fixed baselines before final inference. These are distinct lock stages. The discovery input-boundary repair is documented separately; its archived partial output and exact resume parity remain available.

`results/ENVIRONMENT_AND_CHECKPOINT_AUDIT.json` records actual dependency versions and verifies nine local checkpoint files against the pinned Git-blob/LFS hashes. `audit_results.py` independently checks completed raw-file hashes, question/label alignment, source locks, chronology, token-score sums, fixed-action accuracies and budget accounting. `diagnose_signals.py` adds exploratory paired AUC-difference intervals without fitting or changing any model. `STATISTICAL_APPENDIX.md` exposes validation decisions, corrections/degradations, random-action controls and head overhead.

For CPU-only independent arithmetic inspection of the published complete artifacts, `audit_results.py`, `diagnose_signals.py`, and the two rendering scripts do not import the server-specific affinity bootstrap. They require NumPy, SciPy and (for plots) Matplotlib. Do not rerun `analyze.py train` on a completed locked directory.

PIQA encountered a Windows sharing violation while atomically replacing `STATUS.json`, after 1492 complete result records had been flushed. Its first attempt, exit code1 and full wall-time charge are preserved. `audit_piqa_resume.py` binds recovery to the partial-file hash and verifies unchanged frozen sources; `resume_final.py` runs the remaining questions through the original scorer, including its boundary replay. Monitoring uses a .NET stream with `FileShare.ReadWrite | FileShare.Delete` so that reading progress does not block atomic replacement. Do not use PowerShell `Get-Content` to poll an actively replaced progress file on this host. No scoring rule or test manifest changed for this recovery.

The frozen scorer retains a discovery-specific sentence in `final_piqa_RESUME_PARITY.json.reason` ("Missing-probe handling only"). That sentence is stale for PIQA and is preserved verbatim as emitted; it is not the PIQA failure diagnosis. The actual Windows sharing violation is recorded in `PIQA_OPERATIONAL_RECOVERY.json` and `attempt1_final_piqa.err`. The boundary ID, reused count and numerical parity assertion are valid.

The first optional AUC-uncertainty process exited unsuccessfully with an empty output log when launched without the task bootstrap. Its native failure reason was not captured. The CPU-only recovery uses `report_worker.py` to apply the audited dependency precedence and server CPU affinity; all three report processes then exited0. Primary predictions, locked heads and primary statistical analysis were unchanged. CPU recovery receipts and the failed launch record remain in `logs/`.
