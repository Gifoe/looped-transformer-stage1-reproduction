# Frozen depth-selection results

Fresh RTX5090 / BF16 / torch2.7.1+cu128, 1172 Test questions. Every method shares the same frozen model, prompt, candidate set and scoring protocol. Historical A6000 scores remain separate. All 95% intervals use 10000 paired question bootstraps, seed20261113, without multiplicity correction. The Test set had prior project exposure. No Test-driven refit or parameter revision was performed.

| Method | Correct / 1172 | Accuracy % | vs K4 pp | Rescue / damage | Paired 95% pp CI | Mean loops (replay) |
|---|---:|---:|---:|---:|---|---:|
| K1 | 633 | 54.010 | -33.191 | 44 / 433 | [-36.263, -29.949] | 1.0000 |
| K2 | 954 | 81.399 | -5.802 | 30 / 98 | [-7.679, -3.925] | 2.0000 |
| K3 | 999 | 85.239 | -1.962 | 14 / 37 | [-3.157, -0.768] | 3.0000 |
| K4 | 1022 | 87.201 | +0.000 | 0 / 0 | [+0.000, +0.000] | 4.0000 |
| highest_margin | 1004 | 85.666 | -1.536 | 14 / 32 | [-2.645, -0.427] | 4.0000 |
| highest_entropy | 1013 | 86.433 | -0.768 | 14 / 23 | [-1.792, +0.256] | 4.0000 |
| vote | 1008 | 86.007 | -1.195 | 6 / 20 | [-2.048, -0.341] | 4.0000 |
| stability | 1008 | 86.007 | -1.195 | 6 / 20 | [-2.048, -0.341] | 4.0000 |
| conservative | 1020 | 87.031 | -0.171 | 2 / 4 | [-0.597, +0.256] | 4.0000 |
| rollback | 1022 | 87.201 | +0.000 | 0 / 0 | [+0.000, +0.000] | 4.0000 |

Selecting an earlier endpoint after observing K4 consumes the full4-loop budget.
