# Full-cohort K1–K4 observation map

**Status: OBSERVATION_READY; descriptive only, awaiting independent scientific review.**

All counts are questions; K refers to true native recurrent passes. Whole-cohort test has prior exposure. Full score/activation coverage and same-GPU replay are recorded in results/ROUND_1_OBSERVATION_INDEX.json.

| Split | N | Correct K1/K2/K3/K4 | K1→K2 rescue / damage | K2→K3 rescue / damage | K3→K4 rescue / damage |
|---|---:|---|---:|---:|---:|
| train | 1119 | 609 / 919 / 985 / 990 | 357 / 47 | 82 / 16 | 16 / 11 |
| validation | 299 | 150 / 244 / 257 / 262 | 103 / 9 | 18 / 5 | 8 / 3 |
| test | 1172 | 626 / 953 / 1000 / 1020 | 378 / 51 | 81 / 34 | 33 / 13 |

Test correct trajectory counts (1=correct, 0=wrong, ordered K1–K4):

```json
{"0000": 93, "0001": 13, "0010": 4, "0011": 58, "0100": 12, "0101": 4, "0110": 2, "0111": 360, "1000": 23, "1001": 9, "1010": 5, "1011": 14, "1100": 11, "1101": 7, "1110": 2, "1111": 555}
```

Train includes the 25 fixed demonstrations: excluding them gives n=1094, K1/K2/K3/K4 correct 589/894/960/965, first-transition rescue 352 and damage 47. Validation/test have no demo-ID overlap. Exact top-score ties for test are 37/28/22/18 for K1–K4; first-index argmax is used.

For test readout/post_norm K1→K2 L2 change, per-question group means are rescue 19.737 (n=378), stable correct 19.541 (n=575), damage 17.353 (n=51), stable wrong 17.178 (n=168). These are descriptive means, not evidence of causal localization or separability. Position roles may overlap; recorded representations use current-question selected positions, not complete demonstrations.

## Historical endpoint discrepancy

Historical joint test K1=634 and K4=1021; this run K1=626 and K4=1020. Exactly 28 K1 predictions and 9 K4 predictions differ (36 distinct question IDs). K1: 7 old-wrong→new-correct, 15 old-correct→new-wrong, 6 both wrong with different choices. K4: 3, 4, 2 respectively. The per-ID comparison including request hashes, scores, candidate mappings, targets and runtime is in the indexed historical_joint_comparison.json. All 36 compared questions match historical raw text-request hashes, but historical full token IDs are not available; no cause is established. The validation gate passed its three selected questions and does not establish full-cohort historical numerical identity.

No latent-state causal intervention has been run in this phase. Do not infer mechanism from this map.
