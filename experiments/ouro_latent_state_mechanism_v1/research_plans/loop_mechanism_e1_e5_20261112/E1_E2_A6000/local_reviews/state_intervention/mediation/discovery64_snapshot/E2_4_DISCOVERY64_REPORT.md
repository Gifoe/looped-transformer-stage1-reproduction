# E2.4 full-R × attention: discovery64

Frozen train discovery64 is balanced across four natural outcome groups, not a representative estimate. C00=(x2,a2); C11=(x1,a′); C10=(x1,a2); C01=(x2,a′). a′ is actual attention-localnorm output after only R input x1 is inserted in the natural K2 context, not natural K1 a1. These are controlled mixed states; conclusions are local causal, not an identified natural information path.

## Production predictions by natural group

| Group / condition | N | Correct | Native-correct damaged | Native-wrong rescued | Top ties | Mean gold−allwrong | Mean fixed-wrong | Mean dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all / C00 | 64 | 32 | 0 | 0 | 3 | 2.2835 | 1.6162 | 1.1963 |
| all / C11 | 64 | 34 | 5 | 7 | 2 | 1.1917 | 0.6514 | 0.3799 |
| all / C10 | 64 | 35 | 2 | 5 | 2 | 1.2412 | 0.7168 | 0.4736 |
| all / C01 | 64 | 34 | 0 | 2 | 2 | 2.1735 | 1.4863 | 1.0322 |
| all / full_norm | 64 | 35 | 5 | 8 | 2 | 1.2292 | 0.6582 | 0.3818 |
| all / full_unrelated | 64 | 38 | 0 | 6 | 2 | 1.8870 | 1.3916 | 0.9355 |
| rescue / C00 | 16 | 16 | 0 | 0 | 1 | 3.3477 | 3.1172 | 2.5234 |
| rescue / C11 | 16 | 11 | 5 | 0 | 0 | 1.2956 | 0.8203 | 0.4336 |
| rescue / C10 | 16 | 14 | 2 | 0 | 0 | 1.6237 | 1.2852 | 0.9453 |
| rescue / C01 | 16 | 16 | 0 | 0 | 1 | 2.9661 | 2.5430 | 1.8711 |
| rescue / full_norm | 16 | 11 | 5 | 0 | 0 | 1.3385 | 0.8398 | 0.4375 |
| rescue / full_unrelated | 16 | 16 | 0 | 0 | 0 | 2.5260 | 2.3242 | 1.7422 |
| damage / C00 | 16 | 0 | 0 | 0 | 1 | 0.1549 | -0.7422 | -1.1328 |
| damage / C11 | 16 | 6 | 0 | 6 | 1 | 0.3555 | -0.2031 | -0.3789 |
| damage / C10 | 16 | 4 | 0 | 4 | 2 | 0.2786 | -0.2617 | -0.4844 |
| damage / C01 | 16 | 2 | 0 | 2 | 1 | 0.2083 | -0.6484 | -0.9922 |
| damage / full_norm | 16 | 7 | 0 | 7 | 1 | 0.4128 | -0.1719 | -0.3477 |
| damage / full_unrelated | 16 | 3 | 0 | 3 | 1 | 0.1797 | -0.3438 | -0.8672 |
| stable_correct / C00 | 16 | 16 | 0 | 0 | 0 | 5.1159 | 4.7070 | 4.2539 |
| stable_correct / C11 | 16 | 16 | 0 | 0 | 0 | 2.9063 | 2.6328 | 2.2188 |
| stable_correct / C10 | 16 | 16 | 0 | 0 | 0 | 2.8646 | 2.5586 | 2.2383 |
| stable_correct / C01 | 16 | 16 | 0 | 0 | 0 | 5.0404 | 4.6641 | 4.1289 |
| stable_correct / full_norm | 16 | 16 | 0 | 0 | 0 | 2.9896 | 2.6914 | 2.2852 |
| stable_correct / full_unrelated | 16 | 16 | 0 | 0 | 0 | 4.4349 | 4.0117 | 3.4531 |
| stable_wrong / C00 | 16 | 0 | 0 | 0 | 1 | 0.5156 | -0.6172 | -0.8594 |
| stable_wrong / C11 | 16 | 1 | 0 | 1 | 1 | 0.2096 | -0.6445 | -0.7539 |
| stable_wrong / C10 | 16 | 1 | 0 | 1 | 0 | 0.1979 | -0.7148 | -0.8047 |
| stable_wrong / C01 | 16 | 0 | 0 | 0 | 0 | 0.4792 | -0.6133 | -0.8789 |
| stable_wrong / full_norm | 16 | 1 | 0 | 1 | 1 | 0.1758 | -0.7266 | -0.8477 |
| stable_wrong / full_unrelated | 16 | 3 | 0 | 3 | 1 | 0.4076 | -0.4258 | -0.5859 |
| natural_no_top_tie / C00 | 57 | 30 | 0 | 0 | 0 | 2.4547 | 1.7423 | 1.3300 |
| natural_no_top_tie / C11 | 57 | 32 | 4 | 6 | 2 | 1.3045 | 0.7423 | 0.4550 |
| natural_no_top_tie / C10 | 57 | 32 | 1 | 3 | 2 | 1.3615 | 0.8246 | 0.5636 |
| natural_no_top_tie / C01 | 57 | 31 | 0 | 1 | 1 | 2.3410 | 1.6031 | 1.1458 |
| natural_no_top_tie / full_norm | 57 | 33 | 4 | 7 | 2 | 1.3476 | 0.7500 | 0.4583 |
| natural_no_top_tie / full_unrelated | 57 | 35 | 0 | 5 | 2 | 2.0380 | 1.5296 | 1.0395 |

## Prespecified four-corner contrasts

Attention-at-x2=C00−C01; attention-at-x1=C10−C11; residual-at-a2=C00−C10; residual-at-a′=C01−C11; interaction=C00−C10−C01+C11. The control contrasts use the condition minus named reference. Table shows gold−allwrong contrast; all three metrics, exact BF16 Fraction signs and all IDs are in summary.json.

| Group / contrast | N | Mean | Median | Production −/0/+ | BF16 exact −/0/+ |
|---|---:|---:|---:|---:|---:|
| all / attention_at_x2 | 64 | 0.1100 | 0.0833 | 24/1/39 | 22/4/38 |
| all / attention_at_x1 | 64 | 0.0495 | 0.0417 | 28/0/36 | 28/0/36 |
| all / residual_at_a2 | 64 | 1.0423 | 0.7917 | 14/0/50 | 14/1/49 |
| all / residual_at_aprime | 64 | 0.9818 | 0.7187 | 13/0/51 | 13/1/50 |
| all / interaction | 64 | 0.0605 | 0.0417 | 20/1/43 | 18/9/37 |
| all / fullnorm_vs_C00 | 64 | -1.0544 | -0.8333 | 50/0/14 | 50/0/14 |
| all / unrelated_vs_C00 | 64 | -0.3965 | -0.1250 | 40/0/24 | 37/4/23 |
| all / fullnorm_vs_C11 | 64 | 0.0374 | 0.0417 | 20/4/40 | 18/10/36 |
| all / unrelated_vs_C11 | 64 | 0.6953 | 0.3437 | 24/0/40 | 24/0/40 |
| rescue / attention_at_x2 | 16 | 0.3815 | 0.2083 | 3/0/13 | 3/0/13 |
| rescue / attention_at_x1 | 16 | 0.3281 | 0.2812 | 5/0/11 | 5/0/11 |
| rescue / residual_at_a2 | 16 | 1.7240 | 1.4583 | 0/0/16 | 0/0/16 |
| rescue / residual_at_aprime | 16 | 1.6706 | 1.4687 | 0/0/16 | 0/0/16 |
| rescue / interaction | 16 | 0.0534 | 0.0000 | 7/0/9 | 6/4/6 |
| rescue / fullnorm_vs_C00 | 16 | -2.0091 | -1.7187 | 16/0/0 | 16/0/0 |
| rescue / unrelated_vs_C00 | 16 | -0.8216 | -0.6562 | 14/0/2 | 13/1/2 |
| rescue / fullnorm_vs_C11 | 16 | 0.0430 | 0.0208 | 6/1/9 | 6/2/8 |
| rescue / unrelated_vs_C11 | 16 | 1.2305 | 0.7396 | 5/0/11 | 5/0/11 |
| damage / attention_at_x2 | 16 | -0.0534 | -0.0312 | 10/1/5 | 9/2/5 |
| damage / attention_at_x1 | 16 | -0.0768 | -0.0833 | 9/0/7 | 9/0/7 |
| damage / residual_at_a2 | 16 | -0.1237 | -0.1146 | 10/0/6 | 10/0/6 |
| damage / residual_at_aprime | 16 | -0.1471 | -0.1146 | 9/0/7 | 9/0/7 |
| damage / interaction | 16 | 0.0234 | 0.0417 | 6/0/10 | 6/1/9 |
| damage / fullnorm_vs_C00 | 16 | 0.2578 | 0.1667 | 6/0/10 | 6/0/10 |
| damage / unrelated_vs_C00 | 16 | 0.0247 | 0.0625 | 7/0/9 | 6/1/9 |
| damage / fullnorm_vs_C11 | 16 | 0.0573 | 0.0521 | 2/2/12 | 2/2/12 |
| damage / unrelated_vs_C11 | 16 | -0.1758 | -0.1146 | 9/0/7 | 9/0/7 |
| stable_correct / attention_at_x2 | 16 | 0.0755 | 0.0833 | 5/0/11 | 4/2/10 |
| stable_correct / attention_at_x1 | 16 | -0.0417 | -0.0000 | 8/0/8 | 8/0/8 |
| stable_correct / residual_at_a2 | 16 | 2.2513 | 2.3125 | 0/0/16 | 0/0/16 |
| stable_correct / residual_at_aprime | 16 | 2.1341 | 2.2396 | 0/0/16 | 0/0/16 |
| stable_correct / interaction | 16 | 0.1172 | 0.0521 | 3/0/13 | 3/2/11 |
| stable_correct / fullnorm_vs_C00 | 16 | -2.1263 | -2.2083 | 16/0/0 | 16/0/0 |
| stable_correct / unrelated_vs_C00 | 16 | -0.6810 | -1.2187 | 11/0/5 | 10/1/5 |
| stable_correct / fullnorm_vs_C11 | 16 | 0.0833 | 0.0833 | 3/0/13 | 2/3/11 |
| stable_correct / unrelated_vs_C11 | 16 | 1.5286 | 0.8750 | 4/0/12 | 4/0/12 |
| stable_wrong / attention_at_x2 | 16 | 0.0365 | 0.1042 | 6/0/10 | 6/0/10 |
| stable_wrong / attention_at_x1 | 16 | -0.0117 | 0.0729 | 6/0/10 | 6/0/10 |
| stable_wrong / residual_at_a2 | 16 | 0.3177 | 0.3438 | 4/0/12 | 4/1/11 |
| stable_wrong / residual_at_aprime | 16 | 0.2695 | 0.2708 | 4/0/12 | 4/1/11 |
| stable_wrong / interaction | 16 | 0.0482 | 0.0417 | 4/1/11 | 3/2/11 |
| stable_wrong / fullnorm_vs_C00 | 16 | -0.3398 | -0.3437 | 12/0/4 | 12/0/4 |
| stable_wrong / unrelated_vs_C00 | 16 | -0.1081 | -0.0417 | 8/0/8 | 8/1/7 |
| stable_wrong / fullnorm_vs_C11 | 16 | -0.0339 | -0.0104 | 9/1/6 | 8/3/5 |
| stable_wrong / unrelated_vs_C11 | 16 | 0.1979 | 0.2917 | 6/0/10 | 6/0/10 |
| natural_no_top_tie / attention_at_x2 | 57 | 0.1137 | 0.0833 | 22/0/35 | 20/3/34 |
| natural_no_top_tie / attention_at_x1 | 57 | 0.0570 | 0.0417 | 25/0/32 | 25/0/32 |
| natural_no_top_tie / residual_at_a2 | 57 | 1.0932 | 0.8125 | 11/0/46 | 11/0/46 |
| natural_no_top_tie / residual_at_aprime | 57 | 1.0365 | 0.7500 | 10/0/47 | 10/0/47 |
| natural_no_top_tie / interaction | 57 | 0.0567 | 0.0417 | 20/0/37 | 18/7/32 |
| natural_no_top_tie / fullnorm_vs_C00 | 57 | -1.1071 | -0.8958 | 47/0/10 | 47/0/10 |
| natural_no_top_tie / unrelated_vs_C00 | 57 | -0.4167 | -0.1667 | 36/0/21 | 34/3/20 |
| natural_no_top_tie / fullnorm_vs_C11 | 57 | 0.0431 | 0.0417 | 17/3/37 | 15/9/33 |
| natural_no_top_tie / unrelated_vs_C11 | 57 | 0.7336 | 0.3958 | 20/0/37 | 20/0/37 |

## Decision crossing

All 64: C10 and C01 both correct 31, C10 only 4, C01 only 3, neither 26. Among the five cases correct at C00 but wrong at C11, C10 is correct for three and wrong for two; C01 is correct for all five. The full-norm control damages the same five, while the full-unrelated control damages none of those five. These case counts describe this fixed, stratified sample; the full group-specific crossing is retained below.

| Group | C10/C01 full correctness crossing | C00-correct/C11-wrong subset crossing | Full-control damage crossing |
|---|---|---|---|
| all | {'C10_correct=False;C01_correct=False': 26, 'C10_correct=True;C01_correct=False': 4, 'C10_correct=False;C01_correct=True': 3, 'C10_correct=True;C01_correct=True': 31} | {'C10_correct=True;C01_correct=True': 3, 'C10_correct=False;C01_correct=True': 2} | {'full_norm;C11_damage=False;control_damage=False': 59, 'full_unrelated;C11_damage=False;control_damage=False': 59, 'full_norm;C11_damage=True;control_damage=True': 5, 'full_unrelated;C11_damage=True;control_damage=False': 5} |
| rescue | {'C10_correct=True;C01_correct=True': 14, 'C10_correct=False;C01_correct=True': 2} | {'C10_correct=True;C01_correct=True': 3, 'C10_correct=False;C01_correct=True': 2} | {'full_norm;C11_damage=False;control_damage=False': 11, 'full_unrelated;C11_damage=False;control_damage=False': 11, 'full_norm;C11_damage=True;control_damage=True': 5, 'full_unrelated;C11_damage=True;control_damage=False': 5} |
| damage | {'C10_correct=False;C01_correct=False': 11, 'C10_correct=True;C01_correct=False': 3, 'C10_correct=False;C01_correct=True': 1, 'C10_correct=True;C01_correct=True': 1} | {} | {'full_norm;C11_damage=False;control_damage=False': 16, 'full_unrelated;C11_damage=False;control_damage=False': 16} |
| stable_correct | {'C10_correct=True;C01_correct=True': 16} | {} | {'full_norm;C11_damage=False;control_damage=False': 16, 'full_unrelated;C11_damage=False;control_damage=False': 16} |
| stable_wrong | {'C10_correct=False;C01_correct=False': 15, 'C10_correct=True;C01_correct=False': 1} | {} | {'full_norm;C11_damage=False;control_damage=False': 16, 'full_unrelated;C11_damage=False;control_damage=False': 16} |
| natural_no_top_tie | {'C10_correct=False;C01_correct=False': 23, 'C10_correct=True;C01_correct=False': 3, 'C10_correct=False;C01_correct=True': 2, 'C10_correct=True;C01_correct=True': 29} | {'C10_correct=True;C01_correct=True': 3, 'C10_correct=False;C01_correct=True': 1} | {'full_norm;C11_damage=False;control_damage=False': 53, 'full_unrelated;C11_damage=False;control_damage=False': 53, 'full_norm;C11_damage=True;control_damage=True': 4, 'full_unrelated;C11_damage=True;control_damage=False': 4} |

64 individual candidate scores, gold/fixed/dynamic gaps, ties, correctness and every contrast are in per_question.jsonl; all IDs by sign, groups and exact raw BF16 target-logit signs are in summary.json. CPU control math and independently implemented bare double-hook CUDA full-logit/score replay were completed before analysis. Mixed-state sensitivity, interaction and full-norm effects limit attribution; no explained percentage is claimed.
