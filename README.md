# Training-free and looped Transformer reproduction

**29 October 2026: state-feedback calibration is running; no new success claim.**

The [frozen protocol and code](experiments/closed_loop_value_calibration_20261029/README.md) test whether a tiny value head can turn an intermediate input response into useful continuation, rollback or extrapolation decisions. The backbone remains frozen; these follow-up heads are supervised. The 3,200-question collection and 800-question head-held-out development evaluation have a three-hour cap. Both sets were historically exposed, so this is not independent confirmation. The earlier [400-question controller failed](experiments/closed_loop_action_value_dev400_20261029/FINDINGS_ZH.md): 190/400 versus Adaptive LoopCD 192/400. Historical reports below retain their original scopes.

**28 October2026: prioritized ARC1172 + Qwen600 delivery complete.**

- [1172-question ARC reproduction](experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026/FINDINGS.md): Ouro-1.4B,25-shot; baseline707, Fixed722, Adaptive727 correct. Fixed has positive net15 but exact McNemar p=0.07218, not significant at0.05. [Fresh independent raw-score recheck](experiments/loop_quality_direction_predictability_seed2021_v1/results/ARC1172_DELIVERY_RECHECK.json).
- [600-question Qwen direction-quality report](experiments/loop_quality_direction_predictability_seed2021_v1/FINDINGS.md): Original297, Euler314, nested-CV linear gate314, leave-task-out linear gate313, gold-label Oracle329. **Current selector: No-Go.** Independent new-data validation was deferred by the user and is not claimed.

The new delivery includes all60 features,30,600 policy decisions, nested splits/model-selection records, paired statistics, random controls, five PNG/PDF figures, CPU rebuild code and six passing validation tests. It used zero new GPU hours. [Reproduce](experiments/loop_quality_direction_predictability_seed2021_v1/REPRODUCE.md).

The separate GSM8K generation was interrupted during a server restart at224/3957 records and is incomplete. It was not restarted in this prioritized delivery. Prior global GPU usage, conservatively reconciled through the next boot, is10.073745 hours out of50. This repository is public by user instruction.

The original175 sealed artifacts remain byte-for-byte unchanged. Earlier checkpoint details follow.

Latest completed experiment: full ARC-Challenge, Ouro-1.4B,25-shot, three paper LoopCD configurations.

| Method | Correct /1172 | Accuracy | Corrections | Degradations | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline |707|60.32%|—|—|—|
| Fixed LoopCD |722|61.60%|38|23|+15|
| Adaptive LoopCD |727|62.03%|50|30|+20|

Fixed's paired95% interval is[0.00,2.56]pp and exact McNemar p=0.07218, so this run does not establish a significant Fixed benefit. Adaptive's interval is[0.17,3.24]pp, p=0.03299, unadjusted. The two guidance methods rank differently from the paper.

[Full ARC report](experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026/FINDINGS.md) · [Paired question records](experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026/results/arc_paired_transitions.csv) · [Summary](experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026/results/arc_summary.csv)

The complete raw scores, prompts, code, native exits and independently rebuilt CPU checks are included. The ARC worker charged0.183077 GPU-hours. GSM8K was subsequently interrupted by a system restart; no completed GSM8K claim is made.

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
