# E2.3 pre-frozen discovery64 summary

Exploratory train sample stratified into four natural groups of 16; averages are not representative population estimates. All deltas are condition minus same-input native K2, except forward-add which uses native K1. Negative gold-minus-wrong change means the condition lowered that gap. Production full-vocabulary log-probability contrasts and raw BF16 target-logit exact Fraction signs are tracked separately. No E3 path or mediation is established.

## Condition changes by natural group

| Group / condition | N | Δ gold−mean wrong | Δ fixed K1 wrong | Δ dynamic wrong | Correct | Became wrong / correct | Top ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| all / identity | 64 | 0.0000 | 0.0000 | 0.0000 | 32 | 0 / 0 | 3 |
| all / rollback | 64 | -0.1878 | -0.1514 | -0.1182 | 31 | 2 / 1 | 1 |
| all / norm_matched | 64 | -0.1833 | -0.1426 | -0.1172 | 30 | 2 / 0 | 1 |
| all / unrelated | 64 | -0.0879 | -0.0400 | -0.0479 | 34 | 0 / 2 | 5 |
| all / random_rank4 | 64 | -0.0407 | -0.0127 | -0.0156 | 31 | 1 / 0 | 5 |
| all / rollback_complement | 64 | -0.8939 | -0.8252 | -0.7148 | 36 | 4 / 8 | 2 |
| all / rollback_full | 64 | -1.0918 | -0.9648 | -0.8164 | 34 | 5 / 7 | 2 |
| all / rollback_offered | 64 | -0.0739 | -0.0615 | -0.0498 | 30 | 2 / 0 | 4 |
| all / restore | 64 | 0.0000 | 0.0000 | 0.0000 | 32 | 0 / 0 | 3 |
| all / forward_add | 64 | 0.1637 | 0.0840 | 0.0771 | 33 | 2 / 3 | 1 |
| rescue / identity | 16 | 0.0000 | 0.0000 | 0.0000 | 16 | 0 / 0 | 1 |
| rescue / rollback | 16 | -0.3724 | -0.3789 | -0.3203 | 14 | 2 / 0 | 1 |
| rescue / norm_matched | 16 | -0.3451 | -0.3477 | -0.2930 | 14 | 2 / 0 | 0 |
| rescue / unrelated | 16 | -0.2083 | -0.1914 | -0.1953 | 16 | 0 / 0 | 1 |
| rescue / random_rank4 | 16 | -0.0846 | -0.0937 | -0.0781 | 15 | 1 / 0 | 1 |
| rescue / rollback_complement | 16 | -1.7135 | -1.9805 | -1.8438 | 12 | 4 / 0 | 0 |
| rescue / rollback_full | 16 | -2.0521 | -2.2969 | -2.0898 | 11 | 5 / 0 | 0 |
| rescue / rollback_offered | 16 | -0.1719 | -0.1953 | -0.1602 | 14 | 2 / 0 | 1 |
| rescue / restore | 16 | 0.0000 | 0.0000 | 0.0000 | 16 | 0 / 0 | 1 |
| rescue / forward_add | 16 | 0.2643 | 0.1875 | 0.1758 | 2 | 0 / 2 | 0 |
| damage / identity | 16 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 / 0 | 1 |
| damage / rollback | 16 | 0.0169 | 0.1328 | 0.1484 | 1 | 0 / 1 | 0 |
| damage / norm_matched | 16 | 0.0182 | 0.1367 | 0.1602 | 0 | 0 / 0 | 1 |
| damage / unrelated | 16 | 0.0182 | 0.1328 | 0.0781 | 1 | 0 / 1 | 3 |
| damage / random_rank4 | 16 | 0.0352 | 0.0859 | 0.0859 | 0 | 0 / 0 | 1 |
| damage / rollback_complement | 16 | 0.2135 | 0.4492 | 0.6758 | 6 | 0 / 6 | 1 |
| damage / rollback_full | 16 | 0.2005 | 0.5391 | 0.7539 | 6 | 0 / 6 | 1 |
| damage / rollback_offered | 16 | 0.0260 | 0.0898 | 0.0977 | 0 | 0 / 0 | 3 |
| damage / restore | 16 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 / 0 | 1 |
| damage / forward_add | 16 | 0.0326 | -0.0508 | -0.0508 | 14 | 2 / 0 | 0 |
| stable_correct / identity | 16 | 0.0000 | 0.0000 | 0.0000 | 16 | 0 / 0 | 0 |
| stable_correct / rollback | 16 | -0.3750 | -0.4023 | -0.3594 | 16 | 0 / 0 | 0 |
| stable_correct / norm_matched | 16 | -0.3607 | -0.3711 | -0.3633 | 16 | 0 / 0 | 0 |
| stable_correct / unrelated | 16 | -0.1341 | -0.1445 | -0.1523 | 16 | 0 / 0 | 0 |
| stable_correct / random_rank4 | 16 | -0.1198 | -0.0781 | -0.1133 | 16 | 0 / 0 | 0 |
| stable_correct / rollback_complement | 16 | -1.7734 | -1.6523 | -1.6914 | 16 | 0 / 0 | 0 |
| stable_correct / rollback_full | 16 | -2.2096 | -2.0742 | -2.0352 | 16 | 0 / 0 | 0 |
| stable_correct / rollback_offered | 16 | -0.0833 | -0.1016 | -0.1055 | 16 | 0 / 0 | 0 |
| stable_correct / restore | 16 | 0.0000 | 0.0000 | 0.0000 | 16 | 0 / 0 | 0 |
| stable_correct / forward_add | 16 | 0.3711 | 0.3047 | 0.3047 | 16 | 0 / 0 | 0 |
| stable_wrong / identity | 16 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 / 0 | 1 |
| stable_wrong / rollback | 16 | -0.0208 | 0.0430 | 0.0586 | 0 | 0 / 0 | 0 |
| stable_wrong / norm_matched | 16 | -0.0456 | 0.0117 | 0.0273 | 0 | 0 / 0 | 0 |
| stable_wrong / unrelated | 16 | -0.0273 | 0.0430 | 0.0781 | 1 | 0 / 1 | 1 |
| stable_wrong / random_rank4 | 16 | 0.0065 | 0.0352 | 0.0430 | 0 | 0 / 0 | 3 |
| stable_wrong / rollback_complement | 16 | -0.3021 | -0.1172 | -0.0000 | 2 | 0 / 2 | 1 |
| stable_wrong / rollback_full | 16 | -0.3060 | -0.0273 | 0.1055 | 1 | 0 / 1 | 1 |
| stable_wrong / rollback_offered | 16 | -0.0664 | -0.0391 | -0.0313 | 0 | 0 / 0 | 0 |
| stable_wrong / restore | 16 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 / 0 | 1 |
| stable_wrong / forward_add | 16 | -0.0130 | -0.1055 | -0.1211 | 1 | 0 / 1 | 1 |
| natural_no_top_tie / identity | 57 | 0.0000 | 0.0000 | 0.0000 | 30 | 0 / 0 | 0 |
| natural_no_top_tie / rollback | 57 | -0.1992 | -0.1568 | -0.1261 | 29 | 1 / 0 | 1 |
| natural_no_top_tie / norm_matched | 57 | -0.1930 | -0.1469 | -0.1228 | 29 | 1 / 0 | 0 |
| natural_no_top_tie / unrelated | 57 | -0.0936 | -0.0351 | -0.0559 | 32 | 0 / 2 | 2 |
| natural_no_top_tie / random_rank4 | 57 | -0.0333 | -0.0011 | -0.0110 | 29 | 1 / 0 | 3 |
| natural_no_top_tie / rollback_complement | 57 | -0.9393 | -0.8498 | -0.7664 | 34 | 3 / 7 | 2 |
| natural_no_top_tie / rollback_full | 57 | -1.1502 | -1.0000 | -0.8750 | 32 | 4 / 6 | 2 |
| natural_no_top_tie / rollback_offered | 57 | -0.0618 | -0.0493 | -0.0384 | 29 | 1 / 0 | 3 |
| natural_no_top_tie / restore | 57 | 0.0000 | 0.0000 | 0.0000 | 30 | 0 / 0 | 0 |
| natural_no_top_tie / forward_add | 57 | 0.1747 | 0.0921 | 0.0888 | 31 | 2 / 3 | 1 |

## Paired global-P minus control, gold−mean wrong

Counts list negative / exact-zero / positive. Float signs use production log-probabilities; Fraction signs exactly reconstruct the BF16 target-logit contrast. All paired IDs by sign are in paired_stage_norm_supplement.json.

| Control / group | N | Mean | Median | Production signs −/0/+ | BF16 Fraction signs −/0/+ | P wrong / control correct | P correct / control wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| norm_matched / all | 64 | -0.0046 | 0.0000 | 19/15/30 | 14/30/20 | 0 | 1 |
| norm_matched / rescue | 16 | -0.0273 | -0.0000 | 9/1/6 | 7/5/4 | 0 | 0 |
| norm_matched / damage | 16 | -0.0013 | 0.0000 | 5/1/10 | 2/10/4 | 0 | 1 |
| norm_matched / stable_correct | 16 | -0.0143 | 0.0000 | 5/3/8 | 5/5/6 | 0 | 0 |
| norm_matched / stable_wrong | 16 | 0.0247 | 0.0000 | 0/10/6 | 0/10/6 | 0 | 0 |
| norm_matched / natural_no_top_tie | 57 | -0.0062 | 0.0000 | 18/11/28 | 13/26/18 | 0 | 0 |
| unrelated / all | 64 | -0.0999 | -0.0625 | 38/1/25 | 35/6/23 | 4 | 1 |
| unrelated / rescue | 16 | -0.1641 | -0.1458 | 10/0/6 | 10/0/6 | 2 | 0 |
| unrelated / damage | 16 | -0.0013 | -0.0312 | 10/1/5 | 8/4/4 | 1 | 1 |
| unrelated / stable_correct | 16 | -0.2409 | -0.1875 | 9/0/7 | 9/0/7 | 0 | 0 |
| unrelated / stable_wrong | 16 | 0.0065 | -0.0104 | 9/0/7 | 8/2/6 | 1 | 0 |
| unrelated / natural_no_top_tie | 57 | -0.1056 | -0.0625 | 34/0/23 | 31/5/21 | 3 | 0 |
| random_rank4 / all | 64 | -0.1471 | -0.1667 | 43/2/19 | 43/4/17 | 1 | 1 |
| random_rank4 / rescue | 16 | -0.2878 | -0.2604 | 16/0/0 | 16/0/0 | 1 | 0 |
| random_rank4 / damage | 16 | -0.0182 | 0.0625 | 5/1/10 | 5/2/9 | 0 | 1 |
| random_rank4 / stable_correct | 16 | -0.2552 | -0.2083 | 15/0/1 | 15/0/1 | 0 | 0 |
| random_rank4 / stable_wrong | 16 | -0.0273 | 0.0000 | 7/1/8 | 7/2/7 | 0 | 0 |
| random_rank4 / natural_no_top_tie | 57 | -0.1659 | -0.1667 | 41/1/15 | 41/3/13 | 0 | 0 |
| rollback_complement / all | 64 | 0.7061 | 0.4792 | 15/0/49 | 15/0/49 | 8 | 3 |
| rollback_complement / rescue | 16 | 1.3411 | 0.7083 | 0/0/16 | 0/0/16 | 1 | 3 |
| rollback_complement / damage | 16 | -0.1966 | -0.0938 | 9/0/7 | 9/0/7 | 5 | 0 |
| rollback_complement / stable_correct | 16 | 1.3984 | 1.3229 | 1/0/15 | 1/0/15 | 0 | 0 |
| rollback_complement / stable_wrong | 16 | 0.2812 | 0.1250 | 5/0/11 | 5/0/11 | 2 | 0 |
| rollback_complement / natural_no_top_tie | 57 | 0.7401 | 0.5417 | 11/0/46 | 11/0/46 | 8 | 3 |
| rollback_full / all | 64 | 0.9040 | 0.6042 | 14/0/50 | 14/0/50 | 7 | 4 |
| rollback_full / rescue | 16 | 1.6797 | 1.4167 | 0/0/16 | 0/0/16 | 1 | 4 |
| rollback_full / damage | 16 | -0.1836 | -0.0625 | 10/0/6 | 10/0/6 | 5 | 0 |
| rollback_full / stable_correct | 16 | 1.8346 | 1.8646 | 0/0/16 | 0/0/16 | 0 | 0 |
| rollback_full / stable_wrong | 16 | 0.2852 | 0.2396 | 4/0/12 | 4/0/12 | 1 | 0 |
| rollback_full / natural_no_top_tie | 57 | 0.9510 | 0.6875 | 10/0/47 | 10/0/47 | 7 | 4 |
| rollback_offered / all | 64 | -0.1139 | -0.0833 | 40/0/24 | 38/5/21 | 0 | 1 |
| rollback_offered / rescue | 16 | -0.2005 | -0.2500 | 12/0/4 | 11/2/3 | 0 | 0 |
| rollback_offered / damage | 16 | -0.0091 | -0.0417 | 9/0/7 | 9/0/7 | 0 | 1 |
| rollback_offered / stable_correct | 16 | -0.2917 | -0.3229 | 15/0/1 | 15/0/1 | 0 | 0 |
| rollback_offered / stable_wrong | 16 | 0.0456 | 0.0625 | 4/0/12 | 3/3/10 | 0 | 0 |
| rollback_offered / natural_no_top_tie | 57 | -0.1374 | -0.0833 | 39/0/18 | 37/5/15 | 0 | 0 |

## State-update magnitudes and interaction

| Group | N | Global PΔ L2 | Offered PΔ L2 | Global minus offered L2 | Complement Δ L2 | Offered ranks | Mean full − P − complement outcome change |
|---|---:|---:|---:|---:|---:|---|---:|
| all | 64 | 1.3936 | 0.3203 | 1.3460 | 4.0843 | {'3': 64} | -0.0101 |
| rescue | 16 | 1.3977 | 0.4006 | 1.3287 | 4.2587 | {'3': 16} | 0.0339 |
| damage | 16 | 1.3521 | 0.2689 | 1.3209 | 4.0536 | {'3': 16} | -0.0299 |
| stable_correct | 16 | 1.4865 | 0.3578 | 1.4228 | 4.0105 | {'3': 16} | -0.0612 |
| stable_wrong | 16 | 1.3382 | 0.2540 | 1.3114 | 4.0145 | {'3': 16} | 0.0169 |
| natural_no_top_tie | 57 | 1.3981 | 0.3089 | 1.3548 | 4.0704 | {'3': 57} | -0.0117 |

These means of norms are not fractions of variance or explained natural corrections. Global includes an axis involving token 414, which occurred once as a train candidate and never as train gold; offered-space rollback is a predeclared sensitivity check without a full matched-control family. It does not replace the predeclared global main condition.

Stage Δnorm/cosine are in paired_stage_norm_supplement.json relative to same-input identity-x2 controlled forward: full logits match no-hook native K2 bitwise, but natural intermediate stages were not saved or independently verified. These are descriptive, not mediation. Per-condition norm/cast residual and individual counterexamples/condition scores are in the supplement and per_question.jsonl. Independent control-math audit and full-logit CUDA replay receipts accompany this snapshot. First CPU analysis attempt failed with missing torch import before output creation; corrected script was then run successfully without further inference.
