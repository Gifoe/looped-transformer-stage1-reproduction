# Training-free and looped Transformer reproduction

**Incomplete research checkpoint,26 October2026.** Existing code, raw results,
tables, figures and failure evidence are uploaded here. The remaining formal
experiments and final extension report are still pending.

Read the [current checkpoint](experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/baseline_suite_20261025_seed2021/GITHUB_CHECKPOINT.md)
for the exact completed/pending scope and source limitations.

| Experiment | Measured status |
| --- | --- |
| Three-model multiple-choice baseline/control study | Complete;18 raw files,31,250 readouts; independent MC rebuild passed |
| Ouro1.4B math validation |900/900 records complete |
| Ouro2.6B formal math validation |207/900 records; native failures retained |
| Ouro2.6B isolated affinity diagnostic |216 diagnostic rows, exit0 and independent audit passed; excluded from formal counts |
| Held-out math and actual stopping/latency |Pending |
| Full extension research decision/report/audit |Pending |

The [original sealed Stage1 study](experiments/looped_stage1_trajectory_audit_seed2021_v1/README.md)
and its175-file manifest are preserved byte-for-byte. Its findings describe that
historical protocol. The extension uses disjoint samples and its own protocol.
Direct parity with the official Training-Free Looped Transformer author patch
remains unverified because its repository is inaccessible.

All work uses pretrained BF16 checkpoints with no training. Charged GPU wall time
at this checkpoint is8.685603 hours including Stage1, failures and diagnostics;
the hard total cap is50 hours. Weights, environments, parquet caches, transfer
requests and credentials are excluded. Lightweight prepared requests and tokenizer
exports support independent CPU checks.

From the extension directory, rebuild the completed MC subset with:

```powershell
python -X utf8 scripts/rebuild_cpu.py --scope mc
```

See [reproduction instructions](experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/baseline_suite_20261025_seed2021/REPRODUCE.md).
The default full rebuild requires the outstanding experiments and final research
decision; it does not currently pass. No complete extension or speedup claim is
made by this checkpoint. `CHECKPOINT_MANIFEST.json` records uploaded file hashes.
