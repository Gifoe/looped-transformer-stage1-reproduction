# Frozen causal confirmation of Ouro-1.4B embeddings

[中文复核报告](CAUSAL_CONFIRMATION_ZH.md) · [上一轮完整研究](../ouro_continuous_embedding_20261120/README.md)

This is an extension of the fourth validation round at `bfca32210a0d08d6bc03f070f69e94212ffb88f8`. The same model revision, numerical solvers and trained baselines are held fixed. The primary comparison is FP32 structural-source and rule interventions at K1→K2 on 48 newly generated groups. A preselected 24-group panel carries the same interventions through K2→K3 and K3→K4. Unused ARC validation questions in this embedding series serve as a separate natural-state check. Their original split has been scored in earlier project work.

| Figure | PDF |
| --- | --- |
| 1. Native outcomes at K1→K2 | [Semantic target, other change, unchanged and tie](results/figures/figure_1_early_native_semantic_outcomes.pdf) |
| 2. Counterfactual response prediction | [Structural and dose-matched control arms](results/figures/figure_2_causal_response_controls.pdf) |
| 3. Cross-loop transfer | [Paired structural and ARC natural panels](results/figures/figure_3_paired_loop_transfer.pdf) |

## Reproduce

Use a full checkout on the research branch with the inherited pilot sources and predictor weights. Use the published [Ouro-1.4B Base revision](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1) and the [inherited environment](../ouro_continuous_embedding_20261120/environment.txt).

```bash
cd experiments/ouro_embedding_causal_confirmation_20261120
bash reproduce.sh /absolute/path/to/pinned-model /absolute/path/to/new-run/results
sha256sum -c ARTIFACTS.sha256
```

Freeze, raw candidate outputs, tokenwise response sizes, whole-state errors, measured runtime, branch parity and weight receipts are recorded in [results](results/). The output stores derived measurements and candidate logits. The full activations are used in memory. No model outcomes were used to sample worlds or pick the deeper panel.
