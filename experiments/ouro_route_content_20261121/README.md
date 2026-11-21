# Route × content follow-up in frozen Ouro-1.4B

Anchored to the published native-state study at `9ed73cd0e6568f1b23b4bdb846732b7257def107`. The model snapshot is ByteDance/Ouro-1.4B Base revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

[中文结果](REPORT_ZH.md) · [冻结结构协议](PREREGISTRATION.md) · [自然分量协议](FACTORIAL_PROTOCOL.md) · [早期写入支持](WRITER_SUPPORT_PROTOCOL.md) · [图表](results/figures) · [校验证据](results/RELEASE_VERIFICATION.json)

The four L6 heads were fixed from the earlier label-blind study. The structured intervention crosses two actual donor patterns with two recipient contents while holding the K2 suffix fixed. A stricter version holds the earlier recipient state fixed and crosses only the A and V matrices for those heads. Exact matched-dose row exchange, a separate head group and native query changes provide contrasts. Natural correction is evaluated in fixed, disjoint-within-run OpenBookQA and ARC question cohorts. Results use complete candidate LL and record sum and mean separately. Prompt-only counterfactuals do not restrict model weights or read the gold label.

The raw files contain per-item full-candidate token log likelihoods. `results/NATURAL_MANIFEST.json` is the canonical corrected natural deck. `results/SELECTION_AUDIT.json` records the overlap detected in the preliminary technical run. All reported natural arms, factorials and support tests use the corrected manifest. The structured run used independently generated items.

## Reproduction

Provide the original model directory and the earlier experiment's utilities in the same checkout. Reproduction starts with the byte-preserved published manifests and predictions. The environment and model SHA are in `results/ENVIRONMENT.txt`.

```bash
bash reproduce.sh /path/to/python /path/to/pinned-model /path/to/new-output
sha256sum -c ARTIFACTS.sha256
```

`natural_manifest.py` and `structured_manifest.py` build the manifests from raw snapshots when needed. The official run has separate logs for normal collection and the corrected exclusion deck.

