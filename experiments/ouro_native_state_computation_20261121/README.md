# Frozen Ouro-1.4B natural state computation

Based on `521d71b00d09c8cafafa7541b9db9c43d3341719` with weights `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

- [中文研究报告](REPORT_ZH.md), [主张与证据](CLAIM_EVIDENCE_TABLE.md), [基线](BASELINE_AUDIT.md), and [近邻](NOVELTY_AUDIT.md)
- [原生轨迹](NATIVE_COMPUTATION_ATLAS.md), [Attention](ATTENTION_ROUTING.md), [MLP](MLP_GATING.md), and [有限扰动](FUNCTIONAL_RESPONSE.md)
- [同预算互换](CAUSAL_SWITCH.md), [上游写入](CROSS_LOOP_ORIGIN.md), and [独立结构](GENERALIZATION.md)
- [Figures](results/figures), [paired statistics](results/SUMMARY.json), and [release check](results/RELEASE_VERIFICATION.json)

`BLIND.json` is the gold-free execution manifest and `INPUTS.json` is the corresponding answer-bearing scoring manifest. The development atlas uses only the prompt and ID subset in `ARC_BLIND_INPUTS.json`. Raw `ATLAS_*` records all per-layer and head decomposition statistics. `TRACE_K*` contains a complete representative FP32 trajectory in 16 shards, and `TRACE_VOCAB_LOGITS` holds the fixed answer-position vocabulary readouts.

The same-prompt causal prediction digest and native truth are separated. `CAUSAL_*` uses full-vocabulary alignment and an auxiliary candidate-first-token diagnostic. `FULL_PATCH_*` measures the complete candidate continuations after an actual prompt-restricted intervention. `BEHAVIOR_*` uses native greedy generation and independently scored full answers. `results/formation/` includes the separately locked follow-up that isolates a previous write's role in the next natural factor.

## Reproduction

Use the pinned text-model snapshot directory and the Python/CUDA versions in `results/ENVIRONMENT.txt`. Replays need the inherited `ouro_pair_dynamics_20261120` and `ouro_latent_state_mechanism_v1` directories from the same checkout. The benchmark manifest is published with the code and contains the archived ARC/OpenBookQA input provenance. Replays should use a fresh run directory.

```bash
bash reproduce.sh /path/to/python /path/to/pinned-model /path/to/new-run
python verify.py .
sha256sum -c ARTIFACTS.sha256
```

