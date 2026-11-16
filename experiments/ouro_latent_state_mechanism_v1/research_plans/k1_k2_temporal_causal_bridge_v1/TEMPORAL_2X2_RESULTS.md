# Temporal 2x2: frozen two-loop Ouro-1.4B

Sol performed implementation, execution, analysis and self-audit. No independent agent participated. 32 new development base groups (256 directed units), 64 disjoint confirmation groups (512 units), eight legal text worlds/group. No native-correct filtering. The same underlying text worlds occur in both directions and are not independent observations. All confidence intervals bootstrap base groups 10,000 times with seed 2026111711.

Primary margin is logit(correct W1 label) minus logit(correct W0 label), using the four fixed letter-token candidates. Full-vocabulary log probabilities and exact ties are retained. Prediction uses the first candidate among exact ties.

| Confirmation condition | K2 W1 correct /512 | W0 answer | Other | Exact ties | Mean margin |
| --- | --- | --- | --- | --- | --- |
| C00 | 65 | 441 | 6 | 24 | -1.040649 |
| C01 | 158 | 318 | 36 | 40 | -0.387085 |
| C10 | 318 | 158 | 36 | 40 | 0.387085 |
| C11 | 441 | 65 | 6 | 24 | 1.040649 |

| Contrast | Development ordinary95% | Confirmation ordinary95% | Confirmation Bonferroni2 97.5% |
| --- | --- | --- | --- |
| capacity | 0.571045 [0.511230, 0.630133] | 0.653564 [0.593018, 0.713379] | 0.653564 [0.585448, 0.723633] |
| native | 0.571045 [0.511230, 0.630133] | 0.653564 [0.593018, 0.713379] | 0.653564 [0.585448, 0.723633] |
| K1_prior_contribution | 1.167236 [0.938959, 1.406256] | 1.427734 [1.213620, 1.646240] | 1.427734 [1.189081, 1.680058] |
| interaction | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |

The two primary contrasts passed the frozen adjusted-lower-bound and four mapping/direction sign gates in both phases. Only capacity/native are primary; prior contribution and interaction intervals are descriptive. C01 improves the semantic contrast but has only 158/512 W1 predictions, versus 441/512 natural C11. Delayed delivery is limited capacity, not full recovery of natural reasoning. S itself remains natural W1 in both loops, including C01: only delivery into D is delayed.

**Reciprocal-design identity:** C01 forward equals C10 reverse at both full-vocabulary endpoints, and target-margin sign reverses. Pooled capacity equals pooled native and pooled signed interaction is zero by construction. These are not independent convergent tests and do not establish additive computation. Per-orientation effects, full vectors and absolute interaction distributions are preserved in TEMPORAL_SYMMETRY_AUDIT.json. C00 equals W0 as an isolation check, not scientific discovery. C10 introduces an old reference relation against natural W1 downstream states; loss may include semantic incompatibility. It does not demonstrate lack of K1 answer information.

Natural W1 K1 accuracy 238/512; K2 441/512. K1 wrong→K2 right: 217; right→wrong: 14. These directed-unit counts cover 512 unique confirmation worlds, each represented once as W1. W0 has the same counts by reciprocal balance.

Full baseline-relative margins, all K1/K2 categories and both native baselines are in the phase A JSON summaries. Unique-answer sensitivity, paired non-tie contrasts, and natural-error strata are in TIE_AND_SECONDARY_SENSITIVITY.json. None replaces the all-unit primary analysis.
