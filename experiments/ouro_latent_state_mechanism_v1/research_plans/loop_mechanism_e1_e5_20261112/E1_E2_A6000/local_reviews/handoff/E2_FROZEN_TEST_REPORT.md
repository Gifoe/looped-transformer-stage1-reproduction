# E2 frozen test: full descriptive probe report

One frozen evaluation on 1172 previously exposed ARC Test items; no test-driven model selection. Accuracy is predictive decodability, not causal evidence or proof the original model uses decoded information. Lambda and representations were selected on partially exposed validation. Prior test candidate OOV audit already occurred. unit_l2 removes each input norm before train-fitted coordinate standardization, not a pure angular probe. Label permutation changes 820/1094; two singleton candidate sets cannot shuffle. Native score probabilities below condition full-vocabulary log-probabilities onto offered candidates. Probe logits and native log-probabilities are on different scales; do not compare raw gap magnitudes across them.

## All frozen probes: transfer matrix

| Site / feature / trained loop / control | Lambda | Eval K1 accuracy | Eval K2 accuracy | K1 CE | K2 CE | Probe rescue / damage |
|---|---:|---:|---:|---:|---:|---:|
| l18_input_raw_trainK1_permuted | 100 | 0.2577 | 0.2287 | 1.3855 | 1.3904 | 259 / 293 |
| l18_input_raw_trainK1_real | 1 | 0.4224 | 0.6664 | 1.2540 | 0.9093 | 376 / 90 |
| l18_input_raw_trainK2_permuted | 10000 | 0.2278 | 0.2782 | 1.3859 | 1.3859 | 325 / 266 |
| l18_input_raw_trainK2_real | 1 | 0.4010 | 0.7551 | 1.3068 | 0.6819 | 466 / 51 |
| l18_input_unit_l2_trainK1_permuted | 10000 | 0.2551 | 0.2270 | 1.3859 | 1.3859 | 263 / 296 |
| l18_input_unit_l2_trainK1_real | 1 | 0.4241 | 0.6706 | 1.2534 | 0.9139 | 377 / 88 |
| l18_input_unit_l2_trainK2_permuted | 10000 | 0.2278 | 0.2773 | 1.3859 | 1.3859 | 324 / 266 |
| l18_input_unit_l2_trainK2_real | 1 | 0.4027 | 0.7551 | 1.3057 | 0.6863 | 466 / 53 |
| post_norm_raw_trainK1_permuted | 10000 | 0.1954 | 0.2287 | 1.3859 | 1.3860 | 260 / 221 |
| post_norm_raw_trainK1_real | 1 | 0.5307 | 0.8089 | 1.0895 | 0.5664 | 373 / 47 |
| post_norm_raw_trainK2_permuted | 10000 | 0.2244 | 0.2048 | 1.3859 | 1.3860 | 231 / 254 |
| post_norm_raw_trainK2_real | 1 | 0.5222 | 0.7986 | 1.1197 | 0.5355 | 377 / 53 |
| post_norm_unit_l2_trainK1_permuted | 10000 | 0.2065 | 0.2261 | 1.3859 | 1.3860 | 265 / 242 |
| post_norm_unit_l2_trainK1_real | 1 | 0.5265 | 0.8063 | 1.0947 | 0.5824 | 378 / 50 |
| post_norm_unit_l2_trainK2_permuted | 10000 | 0.2253 | 0.1954 | 1.3860 | 1.3860 | 229 / 264 |
| post_norm_unit_l2_trainK2_real | 1 | 0.5247 | 0.8020 | 1.1151 | 0.5341 | 372 / 47 |

## Native and train-only priors

| Baseline | N | Accuracy | Conditional CE | Brier | ECE | Top ties |
|---|---:|---:|---:|---:|---:|---:|
| native_K1 | 1172 | 0.5341 | 1.1012 | 0.5914 | 0.0256 | 37 |
| native_K2 | 1172 | 0.8131 | 0.5155 | 0.2666 | 0.0347 | 28 |
| prior_uniform | 1172 | 0.2270 | 1.3859 | 0.7498 | 0.0243 | 1172 |
| prior_global | 1172 | 0.2440 | 1.3846 | 0.7494 | 0.0204 | 0 |
| prior_candidate_count | 1172 | 0.2440 | 1.3853 | 0.7495 | 0.0216 | 0 |

## All probe applications: four natural groups and no-tie

Group membership comes from the same historical A6000 native K1/K2 trajectories. Rows show accuracy (N); natural no-tie means no top tie at either natural K1 or K2. Complete per-question predictions, probabilities, candidate values, first tie and paired native groups are retained.

| Probe application | Rescue | Damage | Stable correct | Stable wrong | Natural no tie | Fixed-wrong mean | Dynamic mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| l18_input_raw_trainK1_permuted_evalK1 | 0.235 (378) | 0.333 (51) | 0.268 (575) | 0.250 (168) | 0.255 (1107) | -0.0037 | -0.0387 |
| l18_input_raw_trainK1_permuted_evalK2 | 0.339 (378) | 0.000 (51) | 0.177 (575) | 0.226 (168) | 0.228 (1107) | 0.0027 | -0.0583 |
| l18_input_raw_trainK1_real_evalK1 | 0.238 (378) | 0.490 (51) | 0.616 (575) | 0.155 (168) | 0.426 (1107) | 0.1281 | -0.1168 |
| l18_input_raw_trainK1_real_evalK2 | 0.738 (378) | 0.235 (51) | 0.781 (575) | 0.244 (168) | 0.677 (1107) | 1.1592 | 0.6153 |
| l18_input_raw_trainK2_permuted_evalK1 | 0.339 (378) | 0.000 (51) | 0.174 (575) | 0.232 (168) | 0.229 (1107) | 0.0001 | -0.0011 |
| l18_input_raw_trainK2_permuted_evalK2 | 0.243 (378) | 0.294 (51) | 0.304 (575) | 0.262 (168) | 0.276 (1107) | -0.0002 | -0.0008 |
| l18_input_raw_trainK2_real_evalK1 | 0.320 (378) | 0.333 (51) | 0.517 (575) | 0.208 (168) | 0.409 (1107) | 0.1209 | -0.1739 |
| l18_input_raw_trainK2_real_evalK2 | 0.796 (378) | 0.333 (51) | 0.934 (575) | 0.179 (168) | 0.769 (1107) | 1.7448 | 1.2255 |
| l18_input_unit_l2_trainK1_permuted_evalK1 | 0.238 (378) | 0.294 (51) | 0.257 (575) | 0.274 (168) | 0.253 (1107) | -0.0001 | -0.0006 |
| l18_input_unit_l2_trainK1_permuted_evalK2 | 0.336 (378) | 0.000 (51) | 0.174 (575) | 0.232 (168) | 0.227 (1107) | 0.0001 | -0.0011 |
| l18_input_unit_l2_trainK1_real_evalK1 | 0.246 (378) | 0.490 (51) | 0.614 (575) | 0.155 (168) | 0.426 (1107) | 0.1339 | -0.1212 |
| l18_input_unit_l2_trainK1_real_evalK2 | 0.746 (378) | 0.235 (51) | 0.786 (575) | 0.238 (168) | 0.680 (1107) | 1.1201 | 0.5963 |
| l18_input_unit_l2_trainK2_permuted_evalK1 | 0.339 (378) | 0.000 (51) | 0.174 (575) | 0.232 (168) | 0.229 (1107) | 0.0001 | -0.0012 |
| l18_input_unit_l2_trainK2_permuted_evalK2 | 0.254 (378) | 0.294 (51) | 0.301 (575) | 0.244 (168) | 0.276 (1107) | -0.0002 | -0.0008 |
| l18_input_unit_l2_trainK2_real_evalK1 | 0.310 (378) | 0.333 (51) | 0.525 (575) | 0.214 (168) | 0.411 (1107) | 0.1234 | -0.1748 |
| l18_input_unit_l2_trainK2_real_evalK2 | 0.794 (378) | 0.353 (51) | 0.934 (575) | 0.179 (168) | 0.768 (1107) | 1.7116 | 1.2020 |
| post_norm_raw_trainK1_permuted_evalK1 | 0.257 (378) | 0.216 (51) | 0.136 (575) | 0.256 (168) | 0.190 (1107) | -0.0001 | -0.0006 |
| post_norm_raw_trainK1_permuted_evalK2 | 0.333 (378) | 0.000 (51) | 0.174 (575) | 0.250 (168) | 0.229 (1107) | -0.0001 | -0.0011 |
| post_norm_raw_trainK1_real_evalK1 | 0.204 (378) | 0.647 (51) | 0.864 (575) | 0.089 (168) | 0.544 (1107) | 0.4719 | 0.3275 |
| post_norm_raw_trainK1_real_evalK2 | 0.947 (378) | 0.196 (51) | 0.962 (575) | 0.161 (168) | 0.825 (1107) | 2.6085 | 2.0063 |
| post_norm_raw_trainK2_permuted_evalK1 | 0.331 (378) | 0.000 (51) | 0.174 (575) | 0.226 (168) | 0.226 (1107) | 0.0001 | -0.0011 |
| post_norm_raw_trainK2_permuted_evalK2 | 0.217 (378) | 0.294 (51) | 0.184 (575) | 0.220 (168) | 0.204 (1107) | -0.0003 | -0.0010 |
| post_norm_raw_trainK2_real_evalK1 | 0.217 (378) | 0.588 (51) | 0.838 (575) | 0.107 (168) | 0.538 (1107) | 0.3815 | 0.2163 |
| post_norm_raw_trainK2_real_evalK2 | 0.931 (378) | 0.255 (51) | 0.963 (575) | 0.101 (168) | 0.814 (1107) | 2.7742 | 2.2688 |
| post_norm_unit_l2_trainK1_permuted_evalK1 | 0.243 (378) | 0.235 (51) | 0.163 (575) | 0.262 (168) | 0.201 (1107) | -0.0001 | -0.0007 |
| post_norm_unit_l2_trainK1_permuted_evalK2 | 0.336 (378) | 0.000 (51) | 0.174 (575) | 0.226 (168) | 0.228 (1107) | -0.0000 | -0.0011 |
| post_norm_unit_l2_trainK1_real_evalK1 | 0.204 (378) | 0.627 (51) | 0.854 (575) | 0.101 (168) | 0.538 (1107) | 0.4507 | 0.2790 |
| post_norm_unit_l2_trainK1_real_evalK2 | 0.939 (378) | 0.216 (51) | 0.955 (575) | 0.179 (168) | 0.823 (1107) | 2.3395 | 1.7642 |
| post_norm_unit_l2_trainK2_permuted_evalK1 | 0.336 (378) | 0.000 (51) | 0.170 (575) | 0.232 (168) | 0.227 (1107) | 0.0000 | -0.0011 |
| post_norm_unit_l2_trainK2_permuted_evalK2 | 0.198 (378) | 0.314 (51) | 0.167 (575) | 0.250 (168) | 0.197 (1107) | -0.0003 | -0.0010 |
| post_norm_unit_l2_trainK2_real_evalK1 | 0.225 (378) | 0.549 (51) | 0.843 (575) | 0.101 (168) | 0.542 (1107) | 0.4042 | 0.2362 |
| post_norm_unit_l2_trainK2_real_evalK2 | 0.942 (378) | 0.255 (51) | 0.963 (575) | 0.101 (168) | 0.818 (1107) | 2.6964 | 2.2047 |

## Example counterexamples: fixed K1-trained raw post-norm probe

First ten IDs in sorted order for each class; IDs may overlap between classes only when logically possible. Full candidate-level evidence is in per_question.jsonl.

| Category | IDs |
|---|---|
| native K1 correct but probe K1 wrong (N=96) | ACTAAP_2009_5_8, CSZ20754, CSZ_2009_8_CSZ20740, LEAP_2000_8_2, LEAP_2001_8_10379, LEAP__7_10351, LEAP__8_10365, MCAS_2000_4_10, MCAS_2000_4_30, MCAS_2001_5_3 |
| native K2 correct but probe K2 wrong (N=42) | ACTAAP_2010_7_14, CSZ30169, LEAP_2000_8_2, LEAP__8_10365, MCAS_1998_4_23, MCAS_1999_4_27, MCAS_2000_4_30, MCAS_2005_9_6, Mercury_184100, Mercury_400877 |
| native rescue but probe fails to rescue (N=97) | ACTAAP_2010_7_12, ACTAAP_2010_7_14, AIMS_2008_4_5, AIMS_2008_8_6, CSZ30169, LEAP_2005_8_10404, MCAS_1998_4_23, MCAS_1999_4_26, MCAS_1999_4_27, MCAS_1999_8_16 |
| probe damages K1 to K2 (N=47) | ACTAAP_2007_7_3, AKDE&ED_2008_8_3, FCAT_2008_5_2, LEAP_2011_8_10434, MCAS_2004_8_7, MCAS_2004_9_10-v1, MCAS_2005_9_21, MCAS_2015_5_19, Mercury_177748, Mercury_178255 |

Full conditional CE, Brier, ECE bin counts, all-wrong, fixed-wrong and dynamic margins for all groups appear in summary.json. Cross-loop rescue/damage split by natural group and natural no-tie is in cross_loop_transition_recheck.json. No confidence intervals or p-values were computed.
