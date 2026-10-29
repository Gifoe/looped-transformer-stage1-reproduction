# Statistical appendix

These are the frozen analyses and clearly labeled secondary diagnostics. Final outcomes do not change the selected policy. Primary accuracy is acc_norm.

## Validation and training events

| Training action vs Adaptive | Beneficial | Harmful | Neutral |
|---|---:|---:|---:|
| keep | 92 | 105 | 3003 |
| forward | 45 | 46 | 3109 |
| rollback | 139 | 166 | 2895 |

Events refer to question-action pairs; several actions on one question are not independent questions. Separate action outputs are fitted jointly without treating the outputs as independent test observations.

| Validation policy | Correct /800 | Corrections | Degradations | Net | Delta pp [95% CI] | Holm p |
|---|---:|---:|---:|---:|---|---:|
| adaptive | 367 | 0 | 0 | +0 | +0.000 [0.000, 0.000] | NA |
| keep | 372 | 22 | 17 | +5 | +0.625 [-0.875, 2.125] | NA |
| forward | 371 | 13 | 9 | +4 | +0.500 [-0.625, 1.625] | NA |
| rollback | 364 | 32 | 35 | -3 | -0.375 [-2.375, 1.625] | NA |
| depth1 | 302 | 65 | 130 | -65 | -8.125 [-11.500, -4.750] | NA |
| depth2 | 337 | 49 | 79 | -30 | -3.750 [-6.500, -0.875] | NA |
| negative | 350 | 33 | 50 | -17 | -2.125 [-4.375, 0.125] | NA |
| proxy | 367 | 24 | 24 | +0 | +0.000 [-1.625, 1.750] | 1.0 |
| geometry | 372 | 14 | 9 | +5 | +0.625 [-0.500, 1.875] | 1.0 |
| confidence | 368 | 14 | 13 | +1 | +0.125 [-1.125, 1.375] | 1.0 |
| support | 361 | 23 | 29 | -6 | -0.750 [-2.500, 1.000] | 1.0 |
| format_support | 369 | 15 | 13 | +2 | +0.250 [-1.000, 1.500] | 1.0 |
| ensemble | 352 | 25 | 40 | -15 | -1.875 [-3.875, 0.125] | NA |
| head_controls_accuracy | 368 | 8 | 7 | +1 | +0.125 [-0.750, 1.125] | 1.0 |
| head_controls_logweight | 369 | 3 | 1 | +2 | +0.250 [-0.250, 0.750] | 1.0 |
| head_new_accuracy | 369 | 7 | 5 | +2 | +0.250 [-0.625, 1.125] | 1.0 |
| head_new_logweight | 366 | 2 | 3 | -1 | -0.125 [-0.625, 0.375] | 1.0 |
| head_combined_accuracy | 371 | 9 | 5 | +4 | +0.500 [-0.375, 1.500] | 1.0 |
| head_combined_logweight | 362 | 3 | 8 | -5 | -0.625 [-1.500, 0.125] | 1.0 |
| oracle | 404 | 37 | 0 | +37 | +4.625 [3.250, 6.125] | NA |

Selected: `adaptive`. No candidate passed the precommitted validation gate; retain Adaptive, final comparisons of other policies remain diagnostic.

## final_sciq: paired accuracy

All policies below are compared against Adaptive on the same questions. Only the locked policy has confirmatory status; the others remain diagnostics.

| Policy | Correct | Accuracy % | Corrections | Degradations | Net | Delta pp [95% CI] | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---|---:|
| adaptive | 951 | 95.100 | 0 | 0 | +0 | +0.000 [0.000, 0.000] | 1 |
| keep | 946 | 94.600 | 3 | 8 | -5 | -0.500 [-1.200, 0.100] | 0.226562 |
| forward | 953 | 95.300 | 2 | 0 | +2 | +0.200 [0.000, 0.500] | 0.5 |
| rollback | 944 | 94.400 | 3 | 10 | -7 | -0.700 [-1.400, 0.000] | 0.0922852 |
| depth1 | 523 | 52.300 | 5 | 433 | -428 | -42.800 [-45.900, -39.700] | 3.74242e-121 |
| depth2 | 728 | 72.800 | 6 | 229 | -223 | -22.300 [-25.000, -19.698] | 8.15724e-60 |
| negative | 890 | 89.000 | 5 | 66 | -61 | -6.100 [-7.800, -4.500] | 1.19019e-14 |
| proxy | 947 | 94.700 | 3 | 7 | -4 | -0.400 [-1.000, 0.200] | 0.34375 |
| geometry | 953 | 95.300 | 2 | 0 | +2 | +0.200 [0.000, 0.500] | 0.5 |
| confidence | 954 | 95.400 | 3 | 0 | +3 | +0.300 [0.000, 0.700] | 0.25 |
| support | 949 | 94.900 | 2 | 4 | -2 | -0.200 [-0.700, 0.300] | 0.6875 |
| format_support | 951 | 95.100 | 1 | 1 | +0 | +0.000 [-0.300, 0.300] | 1 |
| ensemble | 947 | 94.700 | 3 | 7 | -4 | -0.400 [-1.000, 0.200] | 0.34375 |
| head_controls_accuracy | 951 | 95.100 | 2 | 2 | +0 | +0.000 [-0.400, 0.400] | 1 |
| head_controls_logweight | 951 | 95.100 | 0 | 0 | +0 | +0.000 [0.000, 0.000] | 1 |
| head_new_accuracy | 951 | 95.100 | 0 | 0 | +0 | +0.000 [0.000, 0.000] | 1 |
| head_new_logweight | 951 | 95.100 | 0 | 0 | +0 | +0.000 [0.000, 0.000] | 1 |
| head_combined_accuracy | 949 | 94.900 | 2 | 4 | -2 | -0.200 [-0.700, 0.300] | 0.6875 |
| head_combined_logweight | 951 | 95.100 | 0 | 0 | +0 | +0.000 [0.000, 0.000] | 1 |
| oracle | 955 | 95.500 | 4 | 0 | +4 | +0.400 [0.100, 0.800] | 0.125 |

### Retaining corrections and avoiding harm

The reference sets below are the beneficial and harmful changes made by Adaptive relative to original Ouro. The selected controller can also change other questions; those effects are captured by the paired table.

| Policy | Benefit retained | Benefit rejected | Harm avoided | Harm accepted |
|---|---:|---:|---:|---:|
| adaptive | 8 | 0 | 0 | 3 |
| keep | 0 | 8 | 3 | 0 |
| forward | 8 | 0 | 2 | 1 |
| rollback | 3 | 5 | 2 | 1 |
| depth1 | 0 | 8 | 1 | 2 |
| depth2 | 1 | 7 | 2 | 1 |
| negative | 0 | 8 | 2 | 1 |
| proxy | 3 | 5 | 3 | 0 |
| geometry | 8 | 0 | 2 | 1 |
| confidence | 8 | 0 | 2 | 1 |
| support | 5 | 3 | 2 | 1 |
| format_support | 7 | 1 | 1 | 2 |
| ensemble | 6 | 2 | 0 | 3 |
| head_controls_accuracy | 6 | 2 | 2 | 1 |
| head_controls_logweight | 8 | 0 | 0 | 3 |
| head_new_accuracy | 8 | 0 | 0 | 3 |
| head_new_logweight | 8 | 0 | 0 | 3 |
| head_combined_accuracy | 5 | 3 | 2 | 1 |
| head_combined_logweight | 8 | 0 | 0 | 3 |

### Action-rate-matched random controls

Permute the observed action assignments across questions, preserving every action count; 1000 shuffles, seed2021. This controls action rates, not conditional question difficulty. The following p values are descriptive and unadjusted.

| Policy | Observed correct | Random mean | Random 95% interval | Randomization p (random >= observed) |
|---|---:|---:|---|---:|
| adaptive | 951 | 951.00 | [951.0, 951.0] | 1.0000 |
| proxy | 947 | 947.26 | [943.0, 951.0] | 0.6533 |
| geometry | 953 | 952.70 | [951.0, 953.0] | 0.7213 |
| confidence | 954 | 951.39 | [950.0, 953.0] | 0.0050 |
| support | 949 | 948.62 | [945.0, 952.0] | 0.5385 |
| format_support | 951 | 951.23 | [949.0, 953.0] | 0.8052 |
| head_controls_accuracy | 951 | 946.66 | [943.0, 950.0] | 0.0160 |
| head_controls_logweight | 951 | 950.96 | [950.0, 952.0] | 0.9211 |
| head_new_accuracy | 951 | 950.27 | [948.0, 952.0] | 0.4575 |
| head_new_logweight | 951 | 950.97 | [950.0, 951.0] | 0.9640 |
| head_combined_accuracy | 949 | 945.99 | [943.0, 950.0] | 0.0879 |
| head_combined_logweight | 951 | 950.94 | [950.0, 952.0] | 0.9161 |

### Update-value discrimination

Only beneficial and harmful (nonzero) events enter AUC; neutral events were retained in training. Confidence-quartile AUC and individual head AUC are in FINAL_ANALYSIS.json. Paired AUC-difference intervals below are exploratory and unadjusted. An interval covering zero does not establish additional information beyond the control head.

| Action | Positive / negative | Head contrast | Delta AUC [95% CI] |
|---|---|---|---|
| keep | 3 / 8 | head_new_accuracy_minus_controls | +0.417 [0.000, 0.833] |
| keep | 3 / 8 | head_combined_accuracy_minus_controls | +0.125 [-0.167, 0.418] |
| keep | 3 / 8 | head_new_logweight_minus_controls | -0.208 [-0.792, 0.375] |
| keep | 3 / 8 | head_combined_logweight_minus_controls | -0.083 [-0.333, 0.125] |
| rollback | 3 / 10 | head_new_accuracy_minus_controls | -0.367 [-1.000, 0.200] |
| rollback | 3 / 10 | head_combined_accuracy_minus_controls | -0.200 [-0.567, 0.100] |
| rollback | 3 / 10 | head_new_logweight_minus_controls | -0.167 [-0.600, 0.200] |
| rollback | 3 / 10 | head_combined_logweight_minus_controls | -0.100 [-0.367, 0.133] |

### Head overhead

CPU wall time includes feature construction, array conversion, scaling and matrix multiply for one head. It excludes backbone passes, which are measured separately.

| Head | Parameters | CPU ms/question |
|---|---:|---:|
| head_controls_accuracy | 96 | 0.6304 |
| head_controls_logweight | 96 | 0.6266 |
| head_new_accuracy | 345 | 0.7463 |
| head_new_logweight | 345 | 0.7439 |
| head_combined_accuracy | 435 | 0.7936 |
| head_combined_logweight | 435 | 0.7919 |

## final_piqa: paired accuracy

All policies below are compared against Adaptive on the same questions. Only the locked policy has confirmatory status; the others remain diagnostics.

| Policy | Correct | Accuracy % | Corrections | Degradations | Net | Delta pp [95% CI] | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---|---:|
| adaptive | 1425 | 79.079 | 0 | 0 | +0 | +0.000 [0.000, 0.000] | 1 |
| keep | 1423 | 78.968 | 40 | 42 | -2 | -0.111 [-1.054, 0.888] | 0.912157 |
| forward | 1422 | 78.912 | 19 | 22 | -3 | -0.166 [-0.832, 0.555] | 0.755229 |
| rollback | 1413 | 78.413 | 55 | 67 | -12 | -0.666 [-1.887, 0.555] | 0.319308 |
| depth1 | 1202 | 66.704 | 86 | 309 | -223 | -12.375 [-14.428, -10.322] | 1.29633e-30 |
| depth2 | 1369 | 75.971 | 86 | 142 | -56 | -3.108 [-4.772, -1.498] | 0.000253055 |
| negative | 1384 | 76.804 | 65 | 106 | -41 | -2.275 [-3.663, -0.832] | 0.00212927 |
| proxy | 1411 | 78.302 | 37 | 51 | -14 | -0.777 [-1.776, 0.222] | 0.165441 |
| geometry | 1423 | 78.968 | 21 | 23 | -2 | -0.111 [-0.832, 0.610] | 0.880396 |
| confidence | 1428 | 79.245 | 27 | 24 | +3 | +0.166 [-0.610, 0.943] | 0.779768 |
| support | 1420 | 78.801 | 50 | 55 | -5 | -0.277 [-1.387, 0.888] | 0.696468 |
| format_support | 1418 | 78.690 | 25 | 32 | -7 | -0.388 [-1.221, 0.444] | 0.427043 |
| ensemble | 1428 | 79.245 | 28 | 25 | +3 | +0.166 [-0.610, 0.943] | 0.783846 |
| head_controls_accuracy | 1425 | 79.079 | 0 | 0 | +0 | +0.000 [0.000, 0.000] | 1 |
| head_controls_logweight | 1423 | 78.968 | 38 | 40 | -2 | -0.111 [-1.054, 0.832] | 0.909946 |
| head_new_accuracy | 1426 | 79.134 | 7 | 6 | +1 | +0.055 [-0.333, 0.444] | 1 |
| head_new_logweight | 1426 | 79.134 | 1 | 0 | +1 | +0.055 [0.000, 0.166] | 1 |
| head_combined_accuracy | 1425 | 79.079 | 0 | 0 | +0 | +0.000 [0.000, 0.000] | 1 |
| head_combined_logweight | 1430 | 79.356 | 38 | 33 | +5 | +0.277 [-0.610, 1.165] | 0.635308 |
| oracle | 1490 | 82.686 | 65 | 0 | +65 | +3.607 [2.775, 4.495] | 5.42101e-20 |

### Retaining corrections and avoiding harm

The reference sets below are the beneficial and harmful changes made by Adaptive relative to original Ouro. The selected controller can also change other questions; those effects are captured by the paired table.

| Policy | Benefit retained | Benefit rejected | Harm avoided | Harm accepted |
|---|---:|---:|---:|---:|
| adaptive | 42 | 0 | 0 | 40 |
| keep | 0 | 42 | 40 | 0 |
| forward | 24 | 18 | 16 | 24 |
| rollback | 11 | 31 | 32 | 8 |
| depth1 | 10 | 32 | 26 | 14 |
| depth2 | 13 | 29 | 32 | 8 |
| negative | 6 | 36 | 33 | 7 |
| proxy | 6 | 36 | 34 | 6 |
| geometry | 24 | 18 | 15 | 25 |
| confidence | 23 | 19 | 24 | 16 |
| support | 10 | 32 | 32 | 8 |
| format_support | 14 | 28 | 22 | 18 |
| ensemble | 32 | 10 | 10 | 30 |
| head_controls_accuracy | 42 | 0 | 0 | 40 |
| head_controls_logweight | 3 | 39 | 38 | 2 |
| head_new_accuracy | 40 | 2 | 6 | 34 |
| head_new_logweight | 42 | 0 | 1 | 39 |
| head_combined_accuracy | 42 | 0 | 0 | 40 |
| head_combined_logweight | 9 | 33 | 36 | 4 |

### Action-rate-matched random controls

Permute the observed action assignments across questions, preserving every action count; 1000 shuffles, seed2021. This controls action rates, not conditional question difficulty. The following p values are descriptive and unadjusted.

| Policy | Observed correct | Random mean | Random 95% interval | Randomization p (random >= observed) |
|---|---:|---:|---|---:|
| adaptive | 1425 | 1425.00 | [1425.0, 1425.0] | 1.0000 |
| proxy | 1411 | 1420.17 | [1411.0, 1429.0] | 0.9840 |
| geometry | 1423 | 1421.77 | [1418.0, 1425.0] | 0.3407 |
| confidence | 1428 | 1422.94 | [1414.0, 1432.0] | 0.1489 |
| support | 1420 | 1419.56 | [1410.0, 1429.0] | 0.5135 |
| format_support | 1418 | 1422.74 | [1413.0, 1432.0] | 0.8641 |
| head_controls_accuracy | 1425 | 1424.90 | [1421.0, 1429.0] | 0.5884 |
| head_controls_logweight | 1423 | 1422.83 | [1420.0, 1426.0] | 0.5854 |
| head_new_accuracy | 1426 | 1423.65 | [1417.0, 1431.025] | 0.2977 |
| head_new_logweight | 1426 | 1424.97 | [1424.0, 1426.0] | 0.1598 |
| head_combined_accuracy | 1425 | 1424.31 | [1419.0, 1430.0] | 0.4695 |
| head_combined_logweight | 1430 | 1422.76 | [1419.0, 1427.0] | 0.0040 |

### Update-value discrimination

Only beneficial and harmful (nonzero) events enter AUC; neutral events were retained in training. Confidence-quartile AUC and individual head AUC are in FINAL_ANALYSIS.json. Paired AUC-difference intervals below are exploratory and unadjusted. An interval covering zero does not establish additional information beyond the control head.

| Action | Positive / negative | Head contrast | Delta AUC [95% CI] |
|---|---|---|---|
| keep | 40 / 42 | head_new_accuracy_minus_controls | +0.232 [0.077, 0.390] |
| keep | 40 / 42 | head_combined_accuracy_minus_controls | +0.110 [0.042, 0.182] |
| keep | 40 / 42 | head_new_logweight_minus_controls | +0.030 [-0.133, 0.183] |
| keep | 40 / 42 | head_combined_logweight_minus_controls | +0.024 [-0.088, 0.130] |
| forward | 19 / 22 | head_new_accuracy_minus_controls | +0.273 [0.060, 0.476] |
| forward | 19 / 22 | head_combined_accuracy_minus_controls | +0.120 [0.029, 0.227] |
| forward | 19 / 22 | head_new_logweight_minus_controls | +0.055 [-0.134, 0.242] |
| forward | 19 / 22 | head_combined_logweight_minus_controls | +0.019 [-0.093, 0.132] |
| rollback | 55 / 67 | head_new_accuracy_minus_controls | +0.129 [-0.021, 0.280] |
| rollback | 55 / 67 | head_combined_accuracy_minus_controls | +0.099 [-0.004, 0.201] |
| rollback | 55 / 67 | head_new_logweight_minus_controls | +0.039 [-0.069, 0.154] |
| rollback | 55 / 67 | head_combined_logweight_minus_controls | +0.012 [-0.054, 0.082] |

### Head overhead

CPU wall time includes feature construction, array conversion, scaling and matrix multiply for one head. It excludes backbone passes, which are measured separately.

| Head | Parameters | CPU ms/question |
|---|---:|---:|
| head_controls_accuracy | 96 | 0.6352 |
| head_controls_logweight | 96 | 0.6330 |
| head_new_accuracy | 345 | 0.7530 |
| head_new_logweight | 345 | 0.7522 |
| head_combined_accuracy | 435 | 0.7999 |
| head_combined_logweight | 435 | 0.8087 |

## Hypothesis interpretation

* **H0:** Failure of the locked gate is compatible with insufficient accessible signal under this representation and sample size; it does not prove that all update-value signals are unidentifiable.
* **H1:** Fixed-sign question-loss, geometry, token-support and formatting policies are falsifiable by their held-out paired accuracy. A favorable post-test diagnostic cannot retroactively pass the validation gate.
* **H2:** The six frozen ridge heads test a bounded family and one training size. Positive AUC alone is insufficient; usable superiority requires the locked controller to pass the paired accuracy gates.
* **H3:** Removing supplied SciQ evidence tests its contribution to this task. The format intervention adds no independent facts. Neither establishes that extra evidence is necessary for every successful frozen controller; H3 remains only partially tested.
