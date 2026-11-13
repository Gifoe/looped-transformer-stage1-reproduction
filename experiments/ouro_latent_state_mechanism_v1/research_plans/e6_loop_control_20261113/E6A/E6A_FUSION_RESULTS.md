# Frozen fusion results

Fresh RTX5090 / BF16 / torch2.7.1+cu128, 1172 Test questions. Every method shares the same frozen model, prompt, candidate set and scoring protocol. Historical A6000 scores remain separate. All 95% intervals use 10000 paired question bootstraps, seed20261113, without multiplicity correction. The Test set had prior project exposure. No Test-driven refit or parameter revision was performed.

| Method | Correct / 1172 | Accuracy % | vs K4 pp | Rescue / damage | Paired 95% pp CI | Mean loops (replay) |
|---|---:|---:|---:|---:|---|---:|
| K1 | 633 | 54.010 | -33.191 | 44 / 433 | [-36.263, -29.949] | 1.0000 |
| K2 | 954 | 81.399 | -5.802 | 30 / 98 | [-7.679, -3.925] | 2.0000 |
| K3 | 999 | 85.239 | -1.962 | 14 / 37 | [-3.157, -0.768] | 3.0000 |
| K4 | 1022 | 87.201 | +0.000 | 0 / 0 | [+0.000, +0.000] | 4.0000 |
| uniform34 | 1018 | 86.860 | -0.341 | 10 / 14 | [-1.195, +0.427] | 4.0000 |
| uniform234 | 1013 | 86.433 | -0.768 | 15 / 24 | [-1.792, +0.256] | 4.0000 |
| uniform1234 | 1011 | 86.263 | -0.939 | 20 / 31 | [-2.133, +0.256] | 4.0000 |
| weighted4 | 1016 | 86.689 | -0.512 | 12 / 18 | [-1.451, +0.427] | 4.0000 |
| probability | 1015 | 86.604 | -0.597 | 15 / 22 | [-1.621, +0.427] | 4.0000 |
| log_probability | 1012 | 86.348 | -0.853 | 15 / 25 | [-1.877, +0.171] | 4.0000 |
| rank | 1001 | 85.410 | -1.792 | 12 / 33 | [-2.901, -0.683] | 4.0000 |
| cal_weight_0012 | 1020 | 87.031 | -0.171 | 10 / 12 | [-0.939, +0.597] | 4.0000 |
| cal_weight_0013 | 1022 | 87.201 | +0.000 | 8 / 8 | [-0.683, +0.683] | 4.0000 |
| cal_weight_0113 | 1019 | 86.945 | -0.256 | 12 / 15 | [-1.109, +0.597] | 4.0000 |
| cal_weight_0124 | 1021 | 87.116 | -0.085 | 13 / 14 | [-0.939, +0.768] | 4.0000 |
| cal_weight_1114 | 1022 | 87.201 | +0.000 | 13 / 13 | [-0.853, +0.853] | 4.0000 |
| cal_weight_1124 | 1024 | 87.372 | +0.171 | 15 / 13 | [-0.683, +1.109] | 4.0000 |
| cal_weight_0122 | 1015 | 86.604 | -0.597 | 14 / 21 | [-1.536, +0.427] | 4.0000 |
| cal_weight_0101 | 1004 | 85.666 | -1.536 | 13 / 31 | [-2.645, -0.427] | 4.0000 |

The declared weighted4 primary does not improve NativeK4. No favorable exploratory Test row is promoted to primary.

Raw uniform234 and log-probability averaging with the same weights are algebraically equivalent. Tiny floating-point perturbations can resolve an exactly/near tied fused winner differently; FUSION_EQUIVALENCE_AUDIT.json lists every discrepancy and score gap. Both frozen outputs are preserved, not treated as two independent method findings.
