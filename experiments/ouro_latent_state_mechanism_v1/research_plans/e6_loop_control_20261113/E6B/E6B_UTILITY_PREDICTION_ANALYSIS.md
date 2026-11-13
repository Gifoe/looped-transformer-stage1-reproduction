# Supervised update-utility diagnostics

| Timing | Features | Alpha | Validation macro-F1 | Test macro-F1 | Test accuracy |
|---|---|---:|---:|---:|---:|
| retrospective | output | 0.1 | 0.5755 | 0.6320 | 0.7648 |
| retrospective | latent | 1.0 | 0.3934 | 0.3964 | 0.5870 |
| retrospective | combined | 0.1 | 0.5681 | 0.6227 | 0.7602 |
| retrospective | unrelated | 1.0 | 0.5792 | 0.6226 | 0.7574 |
| retrospective | shuffled | 0.1 | 0.5783 | 0.6303 | 0.7645 |
| prospective | output | 10.0 | 0.3546 | 0.3825 | 0.6766 |
| prospective | latent | 0.1 | 0.2417 | 0.2762 | 0.5188 |
| prospective | combined | 0.1 | 0.3281 | 0.3740 | 0.6642 |
| prospective | unrelated | 1.0 | 0.3393 | 0.3721 | 0.6591 |
| prospective | shuffled | 0.1 | 0.3278 | 0.3744 | 0.6625 |

Retrospective sees the completed update. Prospective sees only current K2/K3 features and predicts the next transition. Four classes retain stable-wrong, rescue, damage, stable-correct. Train-only standardization and supervised ridge fitting, Validation alpha choice; Test never fitted. Random sample correspondence and matched unrelated-layer controls are independent diagnostic comparators. Predictive signal alone is not zero-label controller utility.

These are fixed ARC project-internal evaluations after historical ARC/Test exposure. Native/outputs/features are same RTX5090 and same score protocol. Bootstrap intervals resample questions, not training seeds; multiple methods and validation choices are disclosed. Depths are causal-prefix policy replay, not measured online latency. No original Ouro weights changed.

Prior mechanism selection used E3 Train1030 and Validation299. Those items are not independent new samples here; historical Test has been exposed repeatedly across earlier project experiments. The new Train1094 excludes25 demonstrations, and its current Train→Validation→frozen Test chronology prevents new-result tuning within this run but does not erase historical selection/exposure. All generalization and interval claims are conditional on this project-internal scope.
