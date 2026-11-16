# Matched baselines and deployment cost

All frozen output-only rules are shown below, including fixed E6 fusion and repository LoopCD. The E6 fixed 0:1:1:2 weighted rule differs from the model-gate weighted output. The strongest Train output rule is E6 conservative (969/1094); no fixed structured row beats it. The strongest observed evaluation baseline is reported descriptively and is not used to select a new policy.

## Validation Joint-Option

All 299 ARC Validation questions; previously exposed and used for checkpoint selection.

| Method | Correct/N | Accuracy | Delta vs native K4 (pp) | Rescue/damage | Paired 95% delta (pp) |
|---|---:|---:|---:|---:|---|
| native_K1 | 148/299 | 49.498% | -37.793 | 8/121 | [-43.813, -31.773] |
| native_K2 | 243/299 | 81.271% | -6.020 | 5/23 | [-9.365, -2.676] |
| native_K3 | 258/299 | 86.288% | -1.003 | 4/7 | [-3.344, +1.003] |
| native_K4 | 261/299 | 87.291% | +0.000 | 0/0 | [+0.000, +0.000] |
| fixedVplus_K2 | 243/299 | 81.271% | -6.020 | 5/23 | [-9.365, -2.676] |
| fixedVplus_K4 | 261/299 | 87.291% | +0.000 | 0/0 | [+0.000, +0.000] |
| catc14_K2 | 243/299 | 81.271% | -6.020 | 5/23 | [-9.365, -2.676] |
| catc14_K4 | 261/299 | 87.291% | +0.000 | 0/0 | [+0.000, +0.000] |
| unstructured14_K2 | 243/299 | 81.271% | -6.020 | 5/23 | [-9.365, -2.676] |
| unstructured14_K4 | 261/299 | 87.291% | +0.000 | 0/0 | [+0.000, +0.000] |
| value7_K2 | 243/299 | 81.271% | -6.020 | 5/23 | [-9.365, -2.676] |
| value7_K4 | 261/299 | 87.291% | +0.000 | 0/0 | [+0.000, +0.000] |
| routing7_K2 | 241/299 | 80.602% | -6.689 | 5/25 | [-10.368, -3.344] |
| routing7_K4 | 261/299 | 87.291% | +0.000 | 0/0 | [+0.000, +0.000] |
| validation_selected_structured_K2 | 241/299 | 80.602% | -6.689 | 5/25 | [-10.368, -3.344] |
| validation_selected_structured_K4 | 261/299 | 87.291% | +0.000 | 0/0 | [+0.000, +0.000] |
| output_logit_mean | 257/299 | 85.953% | -1.338 | 5/9 | [-3.679, +1.003] |
| output_LoopCD_logits_0.25 | 264/299 | 88.294% | +1.003 | 3/0 | [+0.000, +2.341] |
| output_LoopCD_logits_1 | 263/299 | 87.960% | +0.669 | 5/3 | [-1.003, +2.676] |
| output_probability_mean | 258/299 | 86.288% | -1.003 | 5/8 | [-3.344, +1.338] |
| output_max_candidate_confidence | 259/299 | 86.622% | -0.669 | 7/9 | [-3.344, +2.007] |
| output_weighted4 | 262/299 | 87.625% | +0.334 | 1/0 | [+0.000, +1.003] |
| E6_K1 | 148/299 | 49.498% | -37.793 | 8/121 | [-43.813, -31.773] |
| E6_K2 | 243/299 | 81.271% | -6.020 | 5/23 | [-9.365, -2.676] |
| E6_K3 | 258/299 | 86.288% | -1.003 | 4/7 | [-3.344, +1.003] |
| E6_K4 | 261/299 | 87.291% | +0.000 | 0/0 | [+0.000, +0.000] |
| E6_uniform34 | 261/299 | 87.291% | +0.000 | 3/3 | [-1.672, +1.672] |
| E6_uniform234 | 258/299 | 86.288% | -1.003 | 4/7 | [-3.344, +1.003] |
| E6_uniform1234 | 257/299 | 85.953% | -1.338 | 5/9 | [-3.679, +1.003] |
| E6_weighted4 | 258/299 | 86.288% | -1.003 | 3/6 | [-3.010, +1.003] |
| E6_probability | 258/299 | 86.288% | -1.003 | 4/7 | [-3.344, +1.003] |
| E6_log_probability | 258/299 | 86.288% | -1.003 | 4/7 | [-3.344, +1.003] |
| E6_rank | 257/299 | 85.953% | -1.338 | 2/6 | [-3.344, +0.334] |
| E6_highest_margin | 258/299 | 86.288% | -1.003 | 5/8 | [-3.344, +1.338] |
| E6_highest_entropy | 258/299 | 86.288% | -1.003 | 4/7 | [-3.344, +1.003] |
| E6_vote | 259/299 | 86.622% | -0.669 | 2/4 | [-2.341, +1.003] |
| E6_stability | 259/299 | 86.622% | -0.669 | 2/4 | [-2.341, +1.003] |
| E6_conservative | 263/299 | 87.960% | +0.669 | 2/0 | [+0.000, +1.672] |
| E6_rollback | 261/299 | 87.291% | +0.000 | 0/0 | [+0.000, +0.000] |
| E6_early_fixed | 260/299 | 86.957% | -0.334 | 0/1 | [-1.003, +0.000] |
| LoopCD_fixed_0.5 | 262/299 | 87.625% | +0.334 | 3/2 | [-1.003, +1.672] |

## Sciq Joint-Option

997/1000 SciQ Validation questions, after three duplicate-option exclusions fixed without inspecting outcomes. No SciQ examples were used to select coefficients or checkpoints.

| Method | Correct/N | Accuracy | Delta vs native K4 (pp) | Rescue/damage | Paired 95% delta (pp) |
|---|---:|---:|---:|---:|---|
| native_K1 | 724/997 | 72.618% | -20.060 | 20/220 | [-22.869, -17.352] |
| native_K2 | 894/997 | 89.669% | -3.009 | 13/43 | [-4.514, -1.605] |
| native_K3 | 916/997 | 91.876% | -0.802 | 6/14 | [-1.705, +0.100] |
| native_K4 | 924/997 | 92.678% | +0.000 | 0/0 | [+0.000, +0.000] |
| fixedVplus_K2 | 894/997 | 89.669% | -3.009 | 13/43 | [-4.514, -1.605] |
| fixedVplus_K4 | 924/997 | 92.678% | +0.000 | 2/2 | [-0.401, +0.401] |
| catc14_K2 | 894/997 | 89.669% | -3.009 | 13/43 | [-4.514, -1.605] |
| catc14_K4 | 924/997 | 92.678% | +0.000 | 0/0 | [+0.000, +0.000] |
| unstructured14_K2 | 894/997 | 89.669% | -3.009 | 13/43 | [-4.514, -1.605] |
| unstructured14_K4 | 924/997 | 92.678% | +0.000 | 0/0 | [+0.000, +0.000] |
| value7_K2 | 894/997 | 89.669% | -3.009 | 13/43 | [-4.514, -1.605] |
| value7_K4 | 924/997 | 92.678% | +0.000 | 0/0 | [+0.000, +0.000] |
| routing7_K2 | 894/997 | 89.669% | -3.009 | 13/43 | [-4.514, -1.605] |
| routing7_K4 | 922/997 | 92.477% | -0.201 | 1/3 | [-0.602, +0.201] |
| validation_selected_structured_K2 | 894/997 | 89.669% | -3.009 | 13/43 | [-4.514, -1.605] |
| validation_selected_structured_K4 | 922/997 | 92.477% | -0.201 | 1/3 | [-0.602, +0.201] |
| output_logit_mean | 920/997 | 92.277% | -0.401 | 11/15 | [-1.404, +0.602] |
| output_LoopCD_logits_0.25 | 919/997 | 92.177% | -0.502 | 4/9 | [-1.304, +0.201] |
| output_LoopCD_logits_1 | 906/997 | 90.873% | -1.805 | 6/24 | [-2.909, -0.802] |
| output_probability_mean | 919/997 | 92.177% | -0.502 | 10/15 | [-1.505, +0.502] |
| output_max_candidate_confidence | 914/997 | 91.675% | -1.003 | 11/21 | [-2.106, +0.100] |
| output_weighted4 | 921/997 | 92.377% | -0.301 | 3/6 | [-0.903, +0.301] |
| E6_K1 | 724/997 | 72.618% | -20.060 | 20/220 | [-22.869, -17.352] |
| E6_K2 | 894/997 | 89.669% | -3.009 | 13/43 | [-4.514, -1.605] |
| E6_K3 | 916/997 | 91.876% | -0.802 | 6/14 | [-1.705, +0.100] |
| E6_K4 | 924/997 | 92.678% | +0.000 | 0/0 | [+0.000, +0.000] |
| E6_uniform34 | 924/997 | 92.678% | +0.000 | 6/6 | [-0.702, +0.702] |
| E6_uniform234 | 918/997 | 92.076% | -0.602 | 7/13 | [-1.505, +0.301] |
| E6_uniform1234 | 920/997 | 92.277% | -0.401 | 11/15 | [-1.404, +0.602] |
| E6_weighted4 | 917/997 | 91.976% | -0.702 | 5/12 | [-1.505, +0.100] |
| E6_probability | 917/997 | 91.976% | -0.702 | 7/14 | [-1.605, +0.201] |
| E6_log_probability | 918/997 | 92.076% | -0.602 | 7/13 | [-1.505, +0.301] |
| E6_rank | 914/997 | 91.675% | -1.003 | 4/14 | [-1.805, -0.201] |
| E6_highest_margin | 919/997 | 92.177% | -0.502 | 11/16 | [-1.505, +0.502] |
| E6_highest_entropy | 910/997 | 91.274% | -1.404 | 8/22 | [-2.508, -0.401] |
| E6_vote | 916/997 | 91.876% | -0.802 | 3/11 | [-1.605, -0.100] |
| E6_stability | 916/997 | 91.876% | -0.802 | 3/11 | [-1.605, -0.100] |
| E6_conservative | 922/997 | 92.477% | -0.201 | 0/2 | [-0.502, +0.000] |
| E6_rollback | 924/997 | 92.678% | +0.000 | 0/0 | [+0.000, +0.000] |
| E6_early_fixed | 923/997 | 92.578% | -0.100 | 0/1 | [-0.301, +0.000] |
| LoopCD_fixed_0.5 | 914/997 | 91.675% | -1.003 | 4/14 | [-1.906, -0.201] |

## Standard Full-Answer

All 299 ARC Validation questions; matched sequential teacher-forced prefixes, four recurrent passes, no KV cache, 25 full-answer demonstrations. The primary score is the FP32 sum of token log probabilities divided by the answer character count, excluding the leading space. Full token sums and token-normalized alternatives are retained. Native and LoopCD are fresh matched executions; historical batched teacher-forcing numbers are not pooled.

| Method | Correct/N | Accuracy | Delta vs native K4 (pp) | Rescue/damage | Paired 95% delta (pp) |
|---|---:|---:|---:|---:|---|
| nativeK2 | 152/299 | 50.836% | -1.672 | 15/20 | [-5.686, +2.341] |
| nativeK4 | 157/299 | 52.508% | +0.000 | 0/0 | [+0.000, +0.000] |
| LoopCD.25 | 163/299 | 54.515% | +2.007 | 11/5 | [-0.669, +4.682] |
| LoopCD.5 | 169/299 | 56.522% | +4.013 | 19/7 | [+0.669, +7.358] |
| LoopCD1 | 173/299 | 57.860% | +5.351 | 25/9 | [+1.672, +9.365] |
| fixedVplusK4 | 162/299 | 54.181% | +1.672 | 5/0 | [+0.334, +3.344] |
| catc14K4 | 157/299 | 52.508% | +0.000 | 0/0 | [+0.000, +0.000] |
| validation_selected_structuredK4 | 161/299 | 53.846% | +1.338 | 4/0 | [+0.334, +2.676] |

All confidence intervals above are descriptive, question-paired 5,000-bootstrap intervals with seed 20261116. The independent confirmatory family consists of the primary 14-scalar policy and Validation-selected structured policy, each compared with native K4, E6 conservative, and the trained unstructured control (six one-sided Bonferroni contrasts). Both policies fail this benefit gate. Other evaluation populations and score protocols are kept separate.

## Serial deployment profile

Three original technical questions, one warmup and three timed repetitions per question, one full-shaped K4 head, four complete passes. All full-vocabulary endpoints must match the corresponding captured policy exactly. Zero coefficients bypass CATC instrumentation. Nonzero policies retain numerical auditing overhead.

| Policy | Median seconds | Relative to native | Peak allocated GiB |
|---|---:|---:|---:|
| native | 0.1860 | 1.000 | 3.082 |
| fixedVplus | 0.1923 | 1.034 | 3.082 |
| catc14 | 0.1882 | 1.012 | 3.082 |
| unstructured14 | 0.1885 | 1.014 | 3.082 |
| value7 | 0.1891 | 1.017 | 3.082 |
| routing7 | 0.1943 | 1.044 | 3.082 |
| validation_selected_structured | 0.1942 | 1.044 | 3.082 |

The main Phase B capture used four readout heads for native and two for modified policies, so those capture timers are not matched deployment latency. RUNTIME_AND_MEMORY.json preserves those capture costs for exact CPU replay; the serial profiles give the matched inference comparison.
