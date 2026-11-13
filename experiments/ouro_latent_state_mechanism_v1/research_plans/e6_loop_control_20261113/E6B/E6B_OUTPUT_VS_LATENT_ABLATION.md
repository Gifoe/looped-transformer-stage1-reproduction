# Output versus latent matched comparisons

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

## Direct paired latent-minus-output contrasts

| Split | Rule / track | Net correct | Gain pp [95% CI] |
|---|---|---:|---|
| train | rollback / ZL | -1 | -0.091 [-0.548,+0.274] |
| train | agreement / ZL | +4 | +0.366 [-0.366,+1.188] |
| train | stability / ZL | -1 | -0.091 [-0.274,+0.000] |
| train | early_exit / ZL | -1 | -0.091 [-0.274,+0.000] |
| train | rollback / Cal | -1 | -0.091 [-0.274,+0.000] |
| validation | rollback / ZL | +0 | +0.000 [+0.000,+0.000] |
| validation | agreement / ZL | -1 | -0.334 [-2.007,+1.003] |
| validation | stability / ZL | +0 | +0.000 [+0.000,+0.000] |
| validation | early_exit / ZL | +0 | +0.000 [+0.000,+0.000] |
| validation | rollback / Cal | +0 | +0.000 [+0.000,+0.000] |
| test | rollback / ZL | +0 | +0.000 [-0.512,+0.512] |
| test | agreement / ZL | +15 | +1.280 [+0.427,+2.218] |
| test | stability / ZL | +0 | +0.000 [+0.000,+0.000] |
| test | early_exit / ZL | +0 | +0.000 [+0.000,+0.000] |
| test | rollback / Cal | +0 | +0.000 [-0.341,+0.341] |

The agreement rule gains 15 correct decisions from latent gating over its matched output-only rule on Joint Test, while remaining six below native K4. This is an isolated fixed-rule contrast, not evidence of a superior final controller. Rollback ZL and calibrated latent rules have zero net incremental correct decisions over their matched output baselines. Early stopping produces the same Test correct count with more executed loops when the latent gate is enabled.

## Existing frozen E6A confidence/depth controls

All30 paired Test comparisons are retained in E6B_ADDITIONAL_FROZEN_CONTROLS.json. They apply the existing E6A spec directly, with no new rule or Test selection. Primary latent ZL rollback (1018/1172) compares as follows:

| Existing control | Control correct | Latent net correct | Gain pp [95% CI] |
|---|---:|---:|---|
| highest_margin | 1004 | +14 | +1.195 [+0.171,+2.218] |
| highest_entropy | 1013 | +5 | +0.427 [-0.427,+1.280] |
| early_fixed | 1021 | -3 | -0.256 [-0.939,+0.427] |
