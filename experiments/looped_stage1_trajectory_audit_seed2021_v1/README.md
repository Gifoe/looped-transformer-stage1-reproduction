# Looped Transformer stage 1 trajectory audit

Seed: 2021. Frozen checkpoints, BF16, no LLM training. Hard limit: 50 measured
single-GPU process-hours including correctness checks, pilots, and failed runs.

Local project: `D:/chenyu-iclr/experiments/looped_stage1_trajectory_audit_seed2021_v1`.
Server project: `D:/looped_stage1_trajectory_audit_seed2021_v1`.
The local working directory is not a Git repository, so this is an isolated repo.

The requested protocol is retained in `TASK.md`. `configs/protocol.json` fixes
parameters before observing predictions. Model and dataset revisions are recorded
when acquired. Test data never selects strengths or halting thresholds.

## Results and scope

`FINDINGS.md` contains the evidence, all twelve research answers, and the final
Go/No-Go decision. `results/completion_audit.json` checks raw-record counts,
sample alignment, numerical parity and the compute budget. `STATUS.json` records
execution state. Figure 1–5 are provided as PNG and PDF.

The author Qwen patch repository returns HTTP 404, including the final recheck.
The five fixed Qwen configurations therefore use an independently audited recipe.
HF forward/cache identity passes do not establish parity with the missing author
patch. The zero-shot choice scorer is a supplementary protocol, separate from
the original papers' complete benchmark suites. See `ISSUE_LOG.md` and
`sources/IMPLEMENTATION_AUDIT.md` for limitations and repaired failures.

The portable archive retains source, configurations, frozen sample manifests,
all predictions and trajectories, tables, figures and logs. Model weights and
complete source parquet snapshots remain on the server D drive. Their revisions
and SHA-256 metadata are retained in the delivery.

## Rebuild tables and figures

With NumPy, pandas, SciPy and matplotlib installed:

```powershell
python scripts/rebuild_all.py
python -m unittest discover -s tests -v
```

These commands use saved results and require no GPU. `REPRODUCE.md` documents
checkpoint preparation, GPU evaluations, scoring conventions and budget controls.

`scripts/prepare_resources.py` prepares the isolated server environment and
downloads pinned checkpoints through an accessible Hugging Face mirror. It
reserves 5 GiB free space and excludes duplicate weight formats.

SSH credentials are accepted interactively by `scripts/ssh_bridge.py`, held in
memory for that process, and excluded from project files and logs.
