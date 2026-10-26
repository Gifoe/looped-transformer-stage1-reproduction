# Training-free and looped Transformer reproduction

Latest completed experiment: full ARC-Challenge, Ouro-1.4B,25-shot, three paper LoopCD configurations.

| Method | Correct /1172 | Accuracy | Corrections | Degradations | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline |707|60.32%|—|—|—|
| Fixed LoopCD |722|61.60%|38|23|+15|
| Adaptive LoopCD |727|62.03%|50|30|+20|

Fixed's paired95% interval is[0.00,2.56]pp and exact McNemar p=0.07218, so this run does not establish a significant Fixed benefit. Adaptive's interval is[0.17,3.24]pp, p=0.03299, unadjusted. The two guidance methods rank differently from the paper.

[Full ARC report](experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026/FINDINGS.md) · [Paired question records](experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026/results/arc_paired_transitions.csv) · [Summary](experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026/results/arc_summary.csv)

The complete raw scores, prompts, code, native exits and independently rebuilt CPU checks are included. The ARC worker charged0.183077 GPU-hours. GSM8K is running separately; this upload makes no completed GSM8K claim.

Earlier research checkpoint follows:

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
