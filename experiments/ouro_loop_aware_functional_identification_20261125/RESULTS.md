# 已核验的完整历史参照

表格复用 43,769 题的相同 K4 协议。前七项按既有零样本完整候选评分，MMLU-Pro 是真实贪心生成。LoopCD 列是每项主指标上 Fixed 和 Adaptive 中较高的历史得分；末行用未四舍五入的原始比例等权平均。

| Benchmark | n | Native | 最佳 LoopCD | Rank-8 | T1 | T2 | Mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | 1,172 | 52.82% | 54.44% | 67.66% | 68.09% | 64.93% | 56.48% |
| ARC-Easy | 2,376 | 76.68% | 78.83% | 87.37% | 87.71% | 86.32% | 83.12% |
| SciQ | 1,000 | 94.70% | 95.30% | 96.90% | 96.40% | 96.40% | 95.80% |
| MMLU | 14,042 | 68.37% | 68.15% | 67.92% | 67.78% | 67.78% | 66.39% |
| HellaSwag | 10,042 | 71.63% | 72.11% | 76.24% | 76.68% | 74.61% | 71.61% |
| WinoGrande | 1,267 | 66.54% | 66.22% | 67.72% | 66.85% | 66.69% | 66.30% |
| PIQA | 1,838 | 78.56% | 79.11% | 79.82% | 80.14% | 79.54% | 79.27% |
| MMLU-Pro | 12,032 | 26.40% | 27.63% | 37.00% | 36.19% | 37.79% | 21.90% |
| 等权平均 | 43,769 | 66.96% | 67.72% | 72.58% | 72.48% | 71.76% | 67.61% |

同协议固定 LoopCD 平均 67.65%，自适应 LoopCD 平均 67.38%。新选权重及新增四项的完整同协议验证尚未完成。SQuAD 先前的 360 题、GSM8K 的 96 题及之前的局部开发结果均未填入正式结果。 [Mixed、T1 与 T2 已发布逐题结果](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/93b80141e1c81b4a632d4d8a696786c525cde697/experiments/ouro_mixed_full_benchmark_20261124) · [Rank-8 七项候选基线](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/research/ouro-minimal-functional-feedback/experiments/ouro_minimal_functional_feedback_20261123/stage2/zero_shot/REPORT_ZH.md)
