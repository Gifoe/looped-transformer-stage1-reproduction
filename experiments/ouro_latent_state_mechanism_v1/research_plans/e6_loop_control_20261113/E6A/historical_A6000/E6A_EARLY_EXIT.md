# E6-A causal-prefix early-exit simulation

## Historical homogeneous A6000 / Joint-Option Letter Scoring

2590 original questions; development Train excludes25 demonstrations (1094); Validation299 and exposed historical Test1172. All methods use the same archived full candidate scores. This is actual CPU evaluation of recovered native model outputs, not newly collected5090 model inference.

| Method | Track | Correct | vs K4 pp | Rescue/Damage | 95% paired pp CI | Mean loops |
|---|---|---:|---:|---:|---|---:|
| K4 | ZL | 1020/1172 | +0.0000 | 0/0 | [+0.0000, +0.0000] | 4.0000 |
| early_fixed | ZL | 1019/1172 | -0.0853 | 0/1 | [-0.2560, +0.0000] | 3.1015 |
| cal_early_0.6 | Cal | 1015/1172 | -0.4266 | 0/5 | [-0.8532, -0.0853] | 2.8840 |
| cal_early_0.7 | Cal | 1015/1172 | -0.4266 | 0/5 | [-0.8532, -0.0853] | 2.8882 |
| cal_early_0.8 | Cal | 1018/1172 | -0.1706 | 0/2 | [-0.4266, +0.0000] | 2.9258 |
| cal_early_0.9 | Cal | 1019/1172 | -0.0853 | 0/1 | [-0.2560, +0.0000] | 3.0990 |
| cal_early_0.95 | Cal | 1019/1172 | -0.0853 | 0/1 | [-0.2560, +0.0000] | 3.2782 |
| cal_early_0.99 | Cal | 1020/1172 | +0.0000 | 0/0 | [+0.0000, +0.0000] | 3.6920 |

Validation-selected Cal early0.99 preserves1020 correct, with0 rescue/0 damage and simulated3.69198 loops, or92.30% of fixed4 loop count. This is label-calibrated and weight-update-free; it is not zero-label. Latency is NOT MEASURED for this archived simulation and extra intermediate-head costs are not estimated away. No claim of improved actual runtime follows. Pareto summaries are descriptive among this fixed candidate set, not Test-selected deployment policies.
