# Frozen transfer to full-answer scoring

| Track / strategy | Latent | Correct / N | Δ K4 pp [95% CI] | Rescue / damage | Policy mean loops |
|---|---:|---:|---|---:|---:|
| ZL / rollback | False | 702/1172 | -0.256 [-1.024,+0.512] | 9/12 | 4.000 |
| ZL / rollback | True | 700/1172 | -0.427 [-0.939,+0.085] | 2/7 | 4.000 |
| ZL / agreement | False | 697/1172 | -0.683 [-1.877,+0.514] | 23/31 | 4.000 |
| ZL / agreement | True | 701/1172 | -0.341 [-1.109,+0.427] | 8/12 | 4.000 |
| ZL / stability | False | 705/1172 | +0.000 [+0.000,+0.000] | 0/0 | 4.000 |
| ZL / stability | True | 705/1172 | +0.000 [+0.000,+0.000] | 0/0 | 4.000 |
| ZL / early_exit | False | 705/1172 | +0.000 [+0.000,+0.000] | 0/0 | 3.991 |
| ZL / early_exit | True | 705/1172 | +0.000 [+0.000,+0.000] | 0/0 | 3.992 |
| Cal / rollback | False | 702/1172 | -0.256 [-1.024,+0.512] | 9/12 | 4.000 |
| Cal / rollback | True | 700/1172 | -0.427 [-1.024,+0.085] | 3/8 | 4.000 |

Complete answer-text full-vocabulary conditional log likelihood, normalized by answer character count. Same25 demos use full-answer completions. This differs from Joint-option letter scoring; no values are mixed. Joint-frozen rules, thresholds and supervised coefficients transfer with zero refits or FAS parameter changes. The observable R vectors come from fixed candidate0 causal prefix; shape-dependent numeric differences are possible and attested by separate technical native tests.

These are fixed ARC project-internal evaluations after historical ARC/Test exposure. Native/outputs/features are same RTX5090 and same score protocol. Bootstrap intervals resample questions, not training seeds; multiple methods and validation choices are disclosed. Depths are causal-prefix policy replay, not measured online latency. No original Ouro weights changed.

Prior mechanism selection used E3 Train1030 and Validation299. Those items are not independent new samples here; historical Test has been exposed repeatedly across earlier project experiments. The new Train1094 excludes25 demonstrations, and its current Train→Validation→frozen Test chronology prevents new-result tuning within this run but does not erase historical selection/exposure. All generalization and interval claims are conditional on this project-internal scope.

Independent FAS frozen Test: native K4 705/1172; latent ZL rollback 700/1172 (net -5); Joint-selected Cal latent rollback 700/1172 (net -5). Parameters, quantiles and supervised coefficients were transferred without refitting. These outcomes do not replace the Joint endpoint.

## Supervised no-refit diagnostic transfer

| Timing | Features | Test macro-F1 |
|---|---|---:|
| retrospective | output | 0.2223 |
| retrospective | latent | 0.0859 |
| retrospective | combined | 0.2232 |
| retrospective | unrelated | 0.2821 |
| retrospective | shuffled | 0.2269 |
| prospective | output | 0.1414 |
| prospective | latent | 0.0804 |
| prospective | combined | 0.1773 |
| prospective | unrelated | 0.1374 |
| prospective | shuffled | 0.0706 |

## Matched FAS Test latent-minus-output contrasts

| Rule / track | Net correct | Gain pp [95% CI] |
|---|---:|---|
| rollback / ZL | -2 | -0.171 [-0.768,+0.427] |
| agreement / ZL | +4 | +0.341 [-0.683,+1.280] |
| stability / ZL | +0 | +0.000 [+0.000,+0.000] |
| early_exit / ZL | +0 | +0.000 [+0.000,+0.000] |
| rollback / Cal | -2 | -0.171 [-0.683,+0.341] |
