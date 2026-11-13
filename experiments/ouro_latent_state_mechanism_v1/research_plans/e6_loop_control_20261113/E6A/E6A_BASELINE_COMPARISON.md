# Matched baseline comparison

Fresh RTX5090 / BF16 / torch2.7.1+cu128, 1172 Test questions. Every method shares the same frozen model, prompt, candidate set and scoring protocol. Historical A6000 scores remain separate. All 95% intervals use 10000 paired question bootstraps, seed20261113, without multiplicity correction. The Test set had prior project exposure. No Test-driven refit or parameter revision was performed.

| Method | Correct / 1172 | Accuracy % | vs K4 pp | Rescue / damage | Paired 95% pp CI | Mean loops (replay) |
|---|---:|---:|---:|---:|---|---:|
| K1 | 633 | 54.010 | -33.191 | 44 / 433 | [-36.263, -29.949] | 1.0000 |
| K2 | 954 | 81.399 | -5.802 | 30 / 98 | [-7.679, -3.925] | 2.0000 |
| K3 | 999 | 85.239 | -1.962 | 14 / 37 | [-3.157, -0.768] | 3.0000 |
| K4 | 1022 | 87.201 | +0.000 | 0 / 0 | [+0.000, +0.000] | 4.0000 |
| weighted4 | 1016 | 86.689 | -0.512 | 12 / 18 | [-1.451, +0.427] | 4.0000 |
| loopcd_adaptive_cap1 | 1014 | 86.519 | -0.683 | 14 / 22 | [-1.706, +0.256] | 4.0000 |
| loopcd_fixed_0.5 | 1014 | 86.519 | -0.683 | 12 / 20 | [-1.621, +0.256] | 4.0000 |

Repository LoopCD-logits controls: referenceK1, adaptivecap1 and fixedstrength0.5. Adaptive strength uses the full-vocabulary top1/top2 probability gap, not candidate-only confidence; corrected full logits are normalized over the full vocabulary before answer scoring. Cap0 identity passed. Source SHA2563611a505185be5817876581f63b322b1d4c7785d25d5a3078546aba7fbc622bb; original protocol SHA25634041068b2fad3725166406f8360b21ea5cf7e3a91c9edcdabe5f7fd38e13246. This is a matched repository control, not a claim to reproduce every original paper condition. Historical candidate-only archives cannot supply the adaptive gap; that old comparison remains NOT RUN.
