# K1 context reuse

Complete same-device Train1094 (development64 + confirmation1030) and historically exposed Validation299. Every condition keeps R's real attention, MLP and residual computation. Non-R/all, Q/current question-options or P/few-shot prefix are reset to actual final-normalized K1 B at every K2 physical layer input, including L1 where native already equals B. Frozen values pass through the real layer-specific norm, projections, RoPE and causal mask. No original embeddings or future states serve as frozen sources.

Norm controls rescale each B row to that row's current receiver norm. Dose controls feature-permute B-current displacement, preserving per-row FP32 requested L2 dose before BF16 cast; realized dose and norms are recorded per layer and question. Neither restores all compatibility relations in the residual stream. These controls distinguish some magnitude/noise effects, not all hybrid-state incompatibility.


### train1030 (n=1030)

| Condition | Correct | Accuracy | Native correct lost / wrong rescued | Gold-margin delta vs native (interval) |
|---|---:|---:|---:|---|
| K1 | 558/1030 | 54.175% | 337 / 38 | -2.2823 [-2.4103, -2.1539] |
| native | 857/1030 | 83.204% | 0 / 0 | 0.0000 [0.0000, 0.0000] |
| freeze_all | 276/1030 | 26.796% | 619 / 38 | -3.7085 [-3.9050, -3.5186] |
| freeze_Q | 239/1030 | 23.204% | 647 / 29 | -3.5072 [-3.7054, -3.3071] |
| freeze_P | 291/1030 | 28.252% | 603 / 37 | -3.7206 [-3.9147, -3.5280] |
| norm_all | 276/1030 | 26.796% | 620 / 39 | -3.7083 [-3.8652, -3.5491] |
| norm_Q | 238/1030 | 23.107% | 649 / 30 | -3.5090 [-3.6749, -3.3456] |
| norm_P | 293/1030 | 28.447% | 602 / 38 | -3.7201 [-3.8794, -3.5574] |
| dose_all | 288/1030 | 27.961% | 604 / 35 | -4.0635 [-4.2565, -3.8686] |
| dose_Q | 268/1030 | 26.019% | 618 / 29 | -3.9568 [-4.1489, -3.7659] |
| dose_P | 279/1030 | 27.087% | 611 / 33 | -4.0506 [-4.2452, -3.8557] |

### validation299 (n=299)

| Condition | Correct | Accuracy | Native correct lost / wrong rescued | Gold-margin delta vs native (interval) |
|---|---:|---:|---:|---|
| K1 | 148/299 | 49.498% | 104 / 9 | -2.1258 [-2.3616, -1.8825] |
| native | 243/299 | 81.271% | 0 / 0 | 0.0000 [0.0000, 0.0000] |
| freeze_all | 78/299 | 26.087% | 180 / 15 | -3.3495 [-3.7181, -2.9653] |
| freeze_Q | 76/299 | 25.418% | 185 / 18 | -3.0658 [-3.4410, -2.6841] |
| freeze_P | 80/299 | 26.756% | 178 / 15 | -3.3729 [-3.7410, -2.9871] |
| norm_all | 77/299 | 25.753% | 181 / 15 | -3.3479 [-3.6556, -3.0404] |
| norm_Q | 76/299 | 25.418% | 185 / 18 | -3.0681 [-3.3774, -2.7625] |
| norm_P | 79/299 | 26.421% | 178 / 14 | -3.3726 [-3.6835, -3.0569] |
| dose_all | 88/299 | 29.431% | 168 / 13 | -3.4243 [-3.7981, -3.0422] |
| dose_Q | 77/299 | 25.753% | 177 / 11 | -3.5710 [-3.9115, -3.2104] |
| dose_P | 86/299 | 28.763% | 170 / 13 | -3.4435 [-3.8191, -3.0619] |

### development64 (n=64)

| Condition | Correct | Accuracy | Native correct lost / wrong rescued | Gold-margin delta vs native (interval) |
|---|---:|---:|---:|---|
| K1 | 31/64 | 48.438% | 15 / 15 | -1.3086 [-1.9141, -0.6875] |
| native | 31/64 | 48.438% | 0 / 0 | 0.0000 [0.0000, 0.0000] |
| freeze_all | 14/64 | 21.875% | 22 / 5 | -2.3054 [-3.2254, -1.4322] |
| freeze_Q | 16/64 | 25.000% | 22 / 7 | -2.1704 [-3.0186, -1.3812] |
| freeze_P | 20/64 | 31.250% | 20 / 9 | -2.3481 [-3.2704, -1.4549] |
| norm_all | 14/64 | 21.875% | 22 / 5 | -2.3008 [-3.0418, -1.5815] |
| norm_Q | 16/64 | 25.000% | 22 / 7 | -2.1653 [-2.8645, -1.5046] |
| norm_P | 19/64 | 29.688% | 20 / 8 | -2.3511 [-3.1097, -1.6167] |
| dose_all | 24/64 | 37.500% | 18 / 11 | -1.8556 [-2.6664, -1.0356] |
| dose_Q | 23/64 | 35.938% | 21 / 13 | -2.1472 [-2.9947, -1.3220] |
| dose_P | 24/64 | 37.500% | 18 / 11 | -1.8610 [-2.6832, -1.0235] |

### train1094 (n=1094)

| Condition | Correct | Accuracy | Native correct lost / wrong rescued | Gold-margin delta vs native (interval) |
|---|---:|---:|---:|---|
| K1 | 589/1094 | 53.839% | 352 / 53 | -2.2253 [-2.3534, -2.0972] |
| native | 888/1094 | 81.170% | 0 / 0 | 0.0000 [0.0000, 0.0000] |
| freeze_all | 290/1094 | 26.508% | 641 / 43 | -3.6264 [-3.8223, -3.4350] |
| freeze_Q | 255/1094 | 23.309% | 669 / 36 | -3.4290 [-3.6232, -3.2304] |
| freeze_P | 311/1094 | 28.428% | 623 / 46 | -3.6403 [-3.8345, -3.4484] |
| norm_all | 290/1094 | 26.508% | 642 / 44 | -3.6260 [-3.7813, -3.4669] |
| norm_Q | 254/1094 | 23.218% | 671 / 37 | -3.4304 [-3.5865, -3.2668] |
| norm_P | 312/1094 | 28.519% | 622 / 46 | -3.6400 [-3.7963, -3.4783] |
| dose_all | 312/1094 | 28.519% | 622 / 46 | -3.9344 [-4.1264, -3.7425] |
| dose_Q | 291/1094 | 26.600% | 639 / 42 | -3.8510 [-4.0464, -3.6602] |
| dose_P | 303/1094 | 27.697% | 629 / 44 | -3.9225 [-4.1140, -3.7298] |


## Natural corrections and counterexamples

train1030: rescue n=337, all-freeze correct=79; damage n=38, all-freeze correct=5; stable_correct n=520, all-freeze correct=159; stable_wrong n=135, all-freeze correct=33
validation299: rescue n=104, all-freeze correct=29; damage n=9, all-freeze correct=4; stable_correct n=139, all-freeze correct=34; stable_wrong n=47, all-freeze correct=11

The full four-group tables, fixed IDs, every candidate score, tie and individual transitions are in analysis/A_SUMMARY.json and data/A_QUESTION_RESULTS.jsonl.gz. Groups are reconstructed from this device's K1/K2, never imported from historical GPU results. K1 is the genuine first-pass readout, not an E/B hybrid.

Continued R processing can retain a subset of natural corrections from fixed K1 context. The full native K2 behavior is not reproduced by that intervention. Losses establish dependence under this precise frozen-context intervention; they do not uniquely establish a missing semantic operation, because B is reused at layers where native states have different compatibility and scale. Norm and dose results must be read alongside losses. No survival fraction is a universal fraction of reasoning explained.

Intervals use5000 paired question bootstrap replicates, seed20261116. Three primary freeze-vs-native margins use98.333% intervals; other intervals are descriptive95%. Family correction is not global inference across all displayed cohorts/groups/controls. Exposed Validation is not a fresh held-out benchmark.
