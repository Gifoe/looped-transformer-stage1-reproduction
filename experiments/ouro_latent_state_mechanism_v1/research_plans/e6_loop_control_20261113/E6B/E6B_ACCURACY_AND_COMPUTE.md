# Accuracy and policy compute

| Track / strategy | Latent | Correct / N | Δ K4 pp [95% CI] | Rescue / damage | Policy mean loops |
|---|---:|---:|---|---:|---:|
| ZL / rollback | False | 1018/1172 | -0.341 [-1.195,+0.512] | 11/15 | 4.000 |
| ZL / rollback | True | 1018/1172 | -0.341 [-1.024,+0.341] | 6/10 | 4.000 |
| ZL / agreement | False | 1001/1172 | -1.792 [-2.986,-0.597] | 14/35 | 4.000 |
| ZL / agreement | True | 1016/1172 | -0.512 [-1.280,+0.256] | 8/14 | 4.000 |
| ZL / stability | False | 1021/1172 | -0.085 [-0.256,+0.000] | 0/1 | 4.000 |
| ZL / stability | True | 1021/1172 | -0.085 [-0.256,+0.000] | 0/1 | 4.000 |
| ZL / early_exit | False | 1021/1172 | -0.085 [-0.256,+0.000] | 0/1 | 3.105 |
| ZL / early_exit | True | 1021/1172 | -0.085 [-0.256,+0.000] | 0/1 | 3.490 |
| Cal / rollback | False | 1018/1172 | -0.341 [-1.195,+0.512] | 11/15 | 4.000 |
| Cal / rollback | True | 1018/1172 | -0.341 [-1.109,+0.427] | 9/13 | 4.000 |

These are fixed ARC project-internal evaluations after historical ARC/Test exposure. Native/outputs/features are same RTX5090 and same score protocol. Bootstrap intervals resample questions, not training seeds; multiple methods and validation choices are disclosed. Depths are causal-prefix policy replay, not measured online latency. No original Ouro weights changed.

Validation-calibrated gates are separate from ZL; supervised diagnostics are not controller results. LoopCD and both E6A frozen primary paired comparisons are retained in the complete machine-readable summaries. Full-budget rollback/agreement/stability execute four loops; only prefix early_exit has an expected reduced policy depth.

## Actual online Validation resource check

Frozen ZL early-exit p=.9, median Train delta threshold. All observed native prefix vocabulary hashes and final replay predictions/depths matched; actual layer0 executions equal policy depth. Validation299: K4 261 correct, 4 loops, mean .18108 s; output-only 260 correct, 3.12375 loops, .14744 s; latent 260 correct, 3.47492 loops, .16592 s. Latent is approximately 12.5% slower than the matched output-only controller, with equal accuracy. Timing includes callback/head work, and excludes disk export. Alternating method order and warm paired Train3 hooks are recorded in runtime JSON. These are Validation measurements; no online Test or FAS speedup is claimed.

## Actual shared collection resources

These costs include data acquisition work and are separate from controller policy replay. Each Joint item uses one four-loop body and four full-shaped heads; FAS repeats that work for each candidate. Real LoopCD and observed-feature extraction are included. Production windows use synchronized wall-clock timers around forward/scoring/hash work, not isolated kernel timing.

| Protocol / split | N | End-to-end collection s | Production forward/scoring s | Max allocated GiB | R-state storage MiB |
|---|---:|---:|---:|---:|---:|
| joint/train | 1119 | 308.10 | 208.28 | 2.898 | 317.29 |
| joint/validation | 299 | 82.95 | 56.33 | 2.896 | 84.78 |
| joint/test | 1172 | 323.46 | 219.14 | 2.899 | 332.31 |
| full_answer/train | 1119 | 634.64 | 535.46 | 2.823 | 317.39 |
| full_answer/validation | 299 | 169.99 | 143.46 | 2.822 | 84.80 |
| full_answer/test | 1172 | 666.89 | 562.81 | 2.825 | 332.43 |

Model initialization and technical gate overhead are retained in separate logs/receipts rather than represented as online per-question policy speed. CPU memory was not measured; unavailable metrics are marked in runtime JSON.
