# 冻结后缀上的早写与晚写

在 299 道独立 ARC-Easy 上，将第 1 或第 2 圈后的早写传播至下一边界。晚写直接在同一接收状态执行。按每个候选的整条有效序列匹配实际 BF16 写回 RMS，再用同一个原生后缀评分。对照两种原始位置、实际等剂量、Attention 或 MLP 恢复及同计算路径。

| 边界 | 干预 | Δq | 95% 配对区间 | 梯度点积与 Δq r | 范数与 Δq r |
|---|---|---:|---:|---:|---:|
| k1_to_k2 | early | +0.0981 | [+0.079, +0.118] | 0.752 | -0.218 |
| k1_to_k2 | late | +0.1530 | [+0.133, +0.174] | 0.582 | -0.277 |
| k1_to_k2 | early_attention_restored | +0.0777 | [+0.063, +0.093] | 0.783 | -0.246 |
| k1_to_k2 | early_mlp_restored | +0.0913 | [+0.073, +0.110] | 0.641 | -0.252 |
| k1_to_k2 | early_matched | +0.0633 | [+0.050, +0.077] | 0.773 | -0.278 |
| k1_to_k2 | late_matched | +0.1529 | [+0.132, +0.174] | 0.583 | -0.276 |
| k1_to_k2 | early_attention_restored_matched | +0.0787 | [+0.064, +0.094] | 0.674 | -0.293 |
| k1_to_k2 | early_mlp_restored_matched | +0.0642 | [+0.053, +0.076] | 0.750 | -0.282 |

**k1_to_k2 的直接成对比较**。每个候选都符合剂量匹配容差的题数为 299/299。

- `late_minus_early` 的 Δq 为 +0.0550，95% 区间 [+0.042, +0.068]。
- `late_matched_minus_early_matched` 的 Δq 为 +0.0896，95% 区间 [+0.075, +0.105]。
- `early_minus_early_attention_restored` 的 Δq 为 +0.0204，95% 区间 [+0.011, +0.030]。
- `early_matched_minus_early_attention_restored_matched` 的 Δq 为 -0.0154，95% 区间 [-0.024, -0.007]。
- `early_minus_early_mlp_restored` 的 Δq 为 +0.0068，95% 区间 [-0.003, +0.017]。
- `early_matched_minus_early_mlp_restored_matched` 的 Δq 为 -0.0008，95% 区间 [-0.008, +0.007]。

| 边界 | 干预 | Δq | 95% 配对区间 | 梯度点积与 Δq r | 范数与 Δq r |
|---|---|---:|---:|---:|---:|
| k2_to_k3 | early | +0.1530 | [+0.133, +0.174] | 0.742 | -0.249 |
| k2_to_k3 | late | +0.2526 | [+0.225, +0.281] | 0.761 | -0.388 |
| k2_to_k3 | early_attention_restored | +0.1431 | [+0.124, +0.162] | 0.798 | -0.328 |
| k2_to_k3 | early_mlp_restored | +0.2414 | [+0.210, +0.274] | 0.671 | -0.337 |
| k2_to_k3 | early_matched | +0.0921 | [+0.078, +0.106] | 0.886 | -0.306 |
| k2_to_k3 | late_matched | +0.2526 | [+0.225, +0.281] | 0.761 | -0.388 |
| k2_to_k3 | early_attention_restored_matched | +0.1527 | [+0.133, +0.174] | 0.798 | -0.354 |
| k2_to_k3 | early_mlp_restored_matched | +0.1308 | [+0.116, +0.147] | 0.860 | -0.429 |

**k2_to_k3 的直接成对比较**。每个候选都符合剂量匹配容差的题数为 299/299。

- `late_minus_early` 的 Δq 为 +0.0996，95% 区间 [+0.080, +0.120]。
- `late_matched_minus_early_matched` 的 Δq 为 +0.1605，95% 区间 [+0.140, +0.183]。
- `early_minus_early_attention_restored` 的 Δq 为 +0.0100，95% 区间 [-0.001, +0.021]。
- `early_matched_minus_early_attention_restored_matched` 的 Δq 为 -0.0606，95% 区间 [-0.072, -0.049]。
- `early_minus_early_mlp_restored` 的 Δq 为 -0.0884，95% 区间 [-0.112, -0.066]。
- `early_matched_minus_early_mlp_restored_matched` 的 Δq 为 -0.0387，95% 区间 [-0.050, -0.028]。


梯度点积与范数的相关系数是同题统计。恢复 Attention 或 MLP 后重新执行后续模块会包含路径交互，故不能根据该数据给两者作互斥贡献百分比。完整有效序列逐 token 传播保留于 chunks 下。 
[配对和实际计算成本](results/D_TRANSPORT/arc_confirm/TRANSPORT_SUMMARY.json) · [候选原始分数](results/D_TRANSPORT/arc_confirm/PREDICTIONS.jsonl)
