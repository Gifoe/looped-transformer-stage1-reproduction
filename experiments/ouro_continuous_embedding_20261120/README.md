# Continuous-time embedding of Ouro-1.4B

[中文报告](CONTINUOUS_EMBEDDING_ZH.md) · [数学可行性](MATHEMATICAL_FEASIBILITY.md) · [修正方程](MODIFIED_EQUATION.md) · [算子交互](OPERATOR_INTERACTIONS.md) · [连续重建](CONTINUOUS_RECONSTRUCTION.md) · [因果验证](CAUSAL_VALIDATION.md)

This study starts from repository commit `e7b15caa35dda4dbbc75ada5f8a64238ab3a65c0` and reuses the previous native trace and parity implementation. It tests the original discrete update as the target, using the frozen [Ouro-1.4B Base checkpoint](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1). All reconstruction and derivative comparisons use actual model operators. The ridge comparator alone fits a separate endpoint predictor.

| Figure | PDF |
| --- | --- |
| 1. Native and candidate trajectories | [Boundary errors](results/figures/figure_1_embeddings_and_native.pdf) |
| 2. Modified differential equations | [Local error by step size and truncation](results/figures/figure_2_modified_equation.pdf) |
| 3. Attention and MLP order | [Lie bracket and layerwise interventions](results/figures/figure_3_attention_mlp_order.pdf) |
| 4. Errors across 48 branches | [Accumulation and local swap magnitude](results/figures/figure_4_branch_error_sources.pdf) |
| 5. Untouched natural states | [Held-out prediction](results/figures/figure_5_untouched_state_prediction.pdf) |
| 6. Counterfactual responses | [Native and predicted margin change](results/figures/figure_6_counterfactual_responses.pdf) |
| 7. Transfer across loops | [Natural and edited state prediction](results/figures/figure_7_cross_loop.pdf) |

## Reproduce

Clone the full research branch with inherited directories. Use Python 3.12 and [environment.txt](environment.txt). Download all pinned model files. Previous study manifests supply the first dataset; this study deterministically samples and adds new two-hop worlds and layout-locked ARC option permutations. For an ARC intervention, the returned category is also aligned by original option content.

```bash
cd experiments/ouro_continuous_embedding_20261120
bash reproduce.sh /absolute/path/to/pinned-model /absolute/path/to/new-run/results
sha256sum -c ARTIFACTS.sha256
```

The verification command checks the published copy. The reproduction command generates measured errors, directional derivatives, local inverse receipts, full candidate score arrays, tokenwise state/response measures, and timing counters. Full activation tensors are evaluated in memory. The recorded results store the extracted signals instead of every activation tensor.
