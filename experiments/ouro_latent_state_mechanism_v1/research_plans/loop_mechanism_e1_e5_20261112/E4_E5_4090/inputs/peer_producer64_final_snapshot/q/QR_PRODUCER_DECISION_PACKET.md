# Q/R producer decision packet

UTC: 2026-11-12T15:29:43.255467+00:00

Same pre-frozen Train64: 16 each natural rescue/damage/stable-correct/stable-wrong; not population-weighted or blind. Full 64 independent bare model replay passed on each branch. Previously exposed ARC/probe results limit generalization. Both are natural K2 receiver experiments with same-question K1 donor. R patches the R hidden state; Q patches Q-region hidden state, then transfers only its K_Q→L18 R-query attention edge, leaving other SDPA query rows unchanged. Neither establishes the full natural pathway or a writer.

F is gold versus mean non-gold production score, averaged within question and then equally across 16. Each entry below is mean F / number correct. A0K0=clean K2, A0K1=clean carrier with donor K_Q edge, A1K1=whole producer background, A1K0=producer carrier with clean K_Q edge. Index17 is technical target anchor, not upstream.

| Natural group | cut | A0K0 | A0K1 | A1K1 | A1K0 | clean edge loss | producer edge loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| rescue | 0 | 3.348/16 | 2.387/15 | -0.322/3 | -0.313/4 | 0.961 | 0.009 |
| rescue | 8 | 3.348/16 | 2.882/15 | 0.198/4 | 0.654/8 | 0.466 | 0.456 |
| rescue | 16 | 3.348/16 | 2.866/15 | 1.875/14 | 2.430/15 | 0.482 | 0.555 |
| rescue | 17 | 3.348/16 | 2.879/15 | 1.832/13 | 2.457/14 | 0.469 | 0.625 |
| damage | 0 | 0.155/0 | 0.085/1 | 0.430/7 | 0.413/5 | 0.070 | -0.017 |
| damage | 8 | 0.155/0 | 0.258/4 | 0.632/8 | 0.465/8 | -0.103 | -0.167 |
| damage | 16 | 0.155/0 | 0.236/4 | 0.449/6 | 0.280/3 | -0.081 | -0.169 |
| damage | 17 | 0.155/0 | 0.249/4 | 0.421/6 | 0.253/4 | -0.094 | -0.168 |
| stable_correct | 0 | 5.116/16 | 4.258/16 | 0.949/11 | 0.896/11 | 0.858 | -0.053 |
| stable_correct | 8 | 5.116/16 | 4.889/16 | 2.104/14 | 2.349/16 | 0.227 | 0.245 |
| stable_correct | 16 | 5.116/16 | 4.901/16 | 3.969/16 | 4.184/16 | 0.215 | 0.215 |
| stable_correct | 17 | 5.116/16 | 4.905/16 | 3.900/16 | 4.148/16 | 0.211 | 0.249 |
| stable_wrong | 0 | 0.516/0 | 0.359/1 | -0.142/3 | -0.169/1 | 0.156 | -0.027 |
| stable_wrong | 8 | 0.516/0 | 0.404/1 | -0.151/1 | -0.074/1 | 0.112 | 0.077 |
| stable_wrong | 16 | 0.516/0 | 0.388/1 | 0.094/1 | 0.148/1 | 0.128 | 0.055 |
| stable_wrong | 17 | 0.516/0 | 0.389/0 | 0.118/2 | 0.177/1 | 0.126 | 0.059 |

Clean edge loss=F(A0K0)−F(A0K1); producer edge loss=F(A1K0)−F(A1K1). Positive means the respective producer K reduces gap relative to clean K in that background. Full producer loss is F(A0K0)−F(A1K1), not either edge loss.

| Group | cut | R full loss | R correct→wrong / wrong→correct | R target x ΔL2 | Q target x ΔL2 | Q target a ΔL2 |
|---|---:|---:|---:|---:|---:|---:|
| rescue | 0 | 2.395 | 5 / 0 | 8.843 | 6.766 | 1.829 |
| rescue | 8 | 1.038 | 1 / 0 | 3.929 | 3.007 | 1.451 |
| rescue | 16 | 1.999 | 5 / 0 | 4.452 | 0.561 | 1.099 |
| damage | 0 | 0.246 | 0 / 1 | 8.837 | 6.670 | 1.652 |
| damage | 8 | 0.081 | 0 / 2 | 3.903 | 2.531 | 1.205 |
| damage | 16 | -0.180 | 0 / 5 | 4.228 | 0.502 | 0.895 |
| stable_correct | 0 | 3.718 | 4 / 0 | 8.873 | 7.134 | 1.752 |
| stable_correct | 8 | 0.974 | 0 / 0 | 3.895 | 2.743 | 1.244 |
| stable_correct | 16 | 2.156 | 0 / 0 | 4.236 | 0.543 | 0.908 |
| stable_wrong | 0 | 0.072 | 0 / 8 | 8.922 | 6.782 | 1.621 |
| stable_wrong | 8 | 0.031 | 0 / 2 | 4.030 | 2.232 | 0.948 |
| stable_wrong | 16 | 0.281 | 0 / 1 | 4.186 | 0.512 | 0.826 |

Target state distances are unmatched perturbations. They cannot be interpreted as recovered fraction, layer importance, or compared as an amplification factor. Actual R target-anchor and per-question distributions are in the linked R summary.

## Controls, counterexamples, and unresolved questions

- Q rescue: cut0 clean edge loss +0.961 versus producer-background edge +0.009, while whole producer loss is +3.669. For MCAS_2002_8_13 these are +1.458, −0.125, and +5.063; whole producer changes predicted 3→1 while clean edge alone leaves prediction at 3. This is background dependence and bypass potential, not an identified writer.
- Q damage: at cut8 clean edge −0.103 and producer edge −0.167; clean edge gives 4 wrong→correct, producer edge 1 wrong→correct and 1 correct→wrong. All groups and individual reversals remain in rows.
- R rescue cut8 mean loss +1.038 but only 1/16 correct→wrong; cuts0 and16 each give 5/16. Actual cut perturbations are not matched. R also changes Q/K/V and residual routes; it is not a pure query intervention.
- Q index17 is verified against natural K1 K_Q at the target only. Neither upstream Q nor R cuts have same-cut norm-matched or unrelated-direction controls. Prior Q-key norm controls were a different fixed-K1-carrier regional intervention and cannot be substituted here. The 64 Q scores/predictions were independently recomputed by peer; compact rows lack BF16 target logits, so exact Fraction signs are not claimed independently recalculated by that peer.

## Proposed single next discrimination, for decision only

First distinguish R-side L9→L17 candidate recomputation from unmatched perturbation/compensation, while preserving Q-edge bypass as the competing explanation. Freeze before execution one or a small set of actual intermediate boundary states and compare matched-dose R-state rollback/restore versus deterministic unrelated direction at the same cut, with target x_R and final native scores, identity and zero-norm rules. Only then test a predeclared attention-versus-MLP producer if the matched controls support the interval. Do not scan heads, tokens or a broad grid. If Q is preferred, the competitor is a Q-region producer effect mediated specifically by the R-query K edge versus other perturbed-carrier paths; measure both clean and producer backgrounds with the same-cut matched control. These are alternatives for review, not a claim or authorization for either new run.

Freeze exact site/cut/controls and direction before remaining Train non-demo 1030 and validation299; previous ARC/probe outcomes already exposed, so not blind. Preserve negative cases and test no natural-tie sensitivity. Current E1 Train→Val→Test runs under separate frozen forecasts and its later outcomes cannot retroactively choose the E3 target. Q three-case technical 16-condition synchronized wall: 3.49–3.89 s per case excluding some load/save/replay, shared GPU; new conditions require fresh three-case resource measurement.

## Source index

Q: summary SHA 0d62025ba59859904ec808ab6c96400dc594039002a24527b2579bbfad8c9b4b; rows d43063c01cf6886d1108ab3142980710a9a38b206ba784e24f590332d36f8b0a; independent bare audit 034be7fd21b6409ff41291d0fdf79340d6d047c6c50eb7c4cedbeb174519e9ae; target-state norms 81f9eb2242e98c3f44e9ac263fd0058ad063140872f8933e7996dbb1e38ec74f, all in this directory.

R: D:/ouro-e4-e5-20261112/state_intervention/r_producer/discovery64_snapshot: summary SHA 52ac441ff880680dd0c36679460a809471d8bfabb870d089e33bc602ea7f36e8, per_question.jsonl 659ec40ded14801f9808b3c45275d8f2c2252b0fd557302ff0931aa7745709d1, R_PRODUCER_DISCOVERY64_REPORT.md bcc23c32cc3efd3d4e1b7f97f30e09e31bd797db22b23eaf284e7c27075d38ef; independent bare replay c059cc05958ad5195d41671e9d5f682e60621c49620d6032929c306b4f24ca7d. No separate server peer outcomes are combined.
