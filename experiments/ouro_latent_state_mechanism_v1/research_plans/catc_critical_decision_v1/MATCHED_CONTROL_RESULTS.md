# Actual-dose matched controls

All 512 original control questions remain in raw data. One exact development-prompt duplicate was detected during input-hash audit after initial aggregates had been read. The corrected primary population is the other 511 questions; original512 remains separately reported. This retrospective input-only protocol repair does not restore blind confirmation; see COHORT_DUPLICATE_REPAIR.json. Targets are the selected structured policy's cached actual post-BF16 per-site perturbation norms. Random controls use fixed independent Gaussian directions per question and layer; scaling follows the sign of the selected coefficient sum along the current attention output. Both use scalar amplitude search and retain every unmatched dose. No outcome enters the search.

| Method | Correct/N | Accuracy | pp vs native K4 | Rescue/damage vs K4 | Paired 95% delta (pp) | New correct: all native loops wrong |
|---|---:|---:|---:|---:|---|---:|
| native_K1 | 257/511 | 50.294% | -36.986 | 15/204 | [-41.487, -32.290] | 0 |
| native_K2 | 406/511 | 79.452% | -7.828 | 11/51 | [-10.763, -4.892] | 0 |
| native_K3 | 443/511 | 86.693% | -0.587 | 6/9 | [-1.957, +0.783] | 0 |
| native_K4 | 446/511 | 87.280% | +0.000 | 0/0 | [+0.000, +0.000] | 0 |
| selected_K2 | 409/511 | 80.039% | -7.241 | 10/47 | [-10.176, -4.501] | 1 |
| selected_K4 | 446/511 | 87.280% | +0.000 | 0/0 | [+0.000, +0.000] | 0 |
| matched_random_K4 | 449/511 | 87.867% | +0.587 | 3/0 | [+0.000, +1.370] | 0 |
| matched_scaling_K4 | 446/511 | 87.280% | +0.000 | 0/0 | [+0.000, +0.000] | 0 |
| E6_K1 | 257/511 | 50.294% | -36.986 | 15/204 | [-41.487, -32.290] | 0 |
| E6_K2 | 406/511 | 79.452% | -7.828 | 11/51 | [-10.763, -4.892] | 0 |
| E6_K3 | 443/511 | 86.693% | -0.587 | 6/9 | [-1.957, +0.783] | 0 |
| E6_K4 | 446/511 | 87.280% | +0.000 | 0/0 | [+0.000, +0.000] | 0 |
| E6_uniform34 | 447/511 | 87.476% | +0.196 | 3/2 | [-0.587, +0.978] | 0 |
| E6_uniform234 | 439/511 | 85.910% | -1.370 | 7/14 | [-3.131, +0.391] | 0 |
| E6_uniform1234 | 436/511 | 85.323% | -1.957 | 10/20 | [-4.110, +0.000] | 0 |
| E6_weighted4 | 441/511 | 86.301% | -0.978 | 5/10 | [-2.353, +0.391] | 0 |
| E6_probability | 440/511 | 86.106% | -1.174 | 6/12 | [-2.740, +0.391] | 0 |
| E6_log_probability | 441/511 | 86.301% | -0.978 | 7/12 | [-2.740, +0.587] | 0 |
| E6_rank | 439/511 | 85.910% | -1.370 | 4/11 | [-2.935, +0.000] | 0 |
| E6_highest_margin | 439/511 | 85.910% | -1.370 | 7/14 | [-3.131, +0.391] | 0 |
| E6_highest_entropy | 437/511 | 85.519% | -1.761 | 5/14 | [-3.523, -0.196] | 0 |
| E6_vote | 442/511 | 86.497% | -0.783 | 4/8 | [-2.153, +0.587] | 0 |
| E6_stability | 442/511 | 86.497% | -0.783 | 4/8 | [-2.153, +0.587] | 0 |
| E6_conservative | 447/511 | 87.476% | +0.196 | 1/0 | [+0.000, +0.587] | 0 |
| E6_rollback | 446/511 | 87.280% | +0.000 | 0/0 | [+0.000, +0.000] | 0 |
| E6_early_fixed | 446/511 | 87.280% | +0.000 | 0/0 | [+0.000, +0.000] | 0 |
| LoopCD_0.25 | 444/511 | 86.888% | -0.391 | 3/5 | [-1.566, +0.587] | 1 |
| LoopCD_0.5 | 444/511 | 86.888% | -0.391 | 4/6 | [-1.761, +0.783] | 2 |
| LoopCD_1.0 | 445/511 | 87.084% | -0.196 | 5/6 | [-1.566, +0.978] | 3 |

| Comparator | Accuracy difference | Accuracy 95% interval | Bonferroni-3 lower | CE benefit | CE 95% interval | CE Bonferroni-3 lower |
|---|---:|---|---:|---:|---|---:|
| native_K4 | +0.0000000 | [0.0, 0.0] | +0.0000000 | -0.000829165 | [-0.0027454318154433274, 0.0011296388092541297] | -0.002941125 |
| matched_random_K4 | -0.0058708 | [-0.0136986301369863, 0.0] | -0.0136986 | +0.001021386 | [-0.0011462329764663159, 0.0033228139219560553] | -0.001323122 |
| matched_scaling_K4 | +0.0000000 | [0.0, 0.0] | +0.0000000 | +0.000282583 | [-0.0017041138958762833, 0.002299116643651407] | -0.001844392 |

Primary accuracy gate: **False**. Secondary CE-specificity gate: **False**. Each family separately uses three one-sided Bonferroni contrasts (native/random/scaling), 5,000 paired question bootstrap samples, seed 20261116. No global 5% assertion is made across both families. A score gate does not establish final reasoning accuracy. All paired margin contrasts are also retained in CRITICAL_SUMMARY.json.

Matched sites: 7168/7168 within max(1e-9,1% target norm); maximum relative error 0.00761785. Questions with unmatched sites: 0. All such cases remain in the primary analysis. Every actual write obeys the 10% current attention-output cap. DOSE_MATCHING_AUDIT.json lists unmatched sites, and every target/actual dose is retained per question.

Controls are experimental comparisons using companion structured doses, not standalone inference policies. Their measured timing excludes the separately acquired structured reference doses; those references were reused rather than rerun. No outcome-based dose filtering occurred.
