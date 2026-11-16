# Actual G→later attention path: fixed coarse intervals

Frozen G=entire map region. H=all later query/options/formatting/read positions. Z uses W0 S KV at all K2 D receivers. T opens S→G only. On T trajectories, block H reads of G using actual Z post-RoPE same-layer-input G K/V; G states and current Q/current remaining D KV remain live. No hidden states or answer logits are copied. Actual G KV equals T and native W1 at every layer, since later H cannot causally influence G.

Three intervals were frozen before any C outcomes: L1–8, L9–16, L17–24. Restore allows current G KV only within that interval. Reverse blocks only that interval. The seven primary contrasts use ordinary95% and two-sided Bonferroni7 99.2857% intervals. Claim an interval only if restore and reverse-block pass adjusted lower>0 and all four map/direction signs in both phases. All intervals retained, no confirmation selection or finer search.

| Contrast | Dev mean | Confirmation ordinary95% | Confirmation Bonferroni7 | Both-phase individual gate |
| --- | --- | --- | --- | --- |
| restore_early_vs_block | 0.039795 | 0.034424 [0.024658, 0.044556] | 0.034424 [0.021083, 0.048253] | True |
| T_vs_reverse_early | 0.028809 | 0.044189 [0.034424, 0.053955] | 0.044189 [0.030604, 0.057861] | True |
| restore_mid_vs_block | 0.506348 | 0.593506 [0.537228, 0.649292] | 0.593506 [0.515432, 0.671249] | True |
| T_vs_reverse_mid | 0.494873 | 0.593018 [0.537842, 0.648196] | 0.593018 [0.516322, 0.668859] | True |
| restore_late_vs_block | 0.004395 | -0.004883 [-0.011841, 0.001953] | -0.004883 [-0.014648, 0.004395] | False |
| T_vs_reverse_late | -0.001465 | -0.000244 [-0.007202, 0.006714] | -0.000244 [-0.009923, 0.008946] | False |
| T_vs_matched_random | 0.529053 | 0.647461 [0.582639, 0.711792] | 0.647461 [0.558922, 0.735997] | True |

| Confirmation condition | K2 W1 /512 | Mean margin | Ties |
| --- | --- | --- | --- |
| Z | 318 | 0.387085 | 40 |
| T | 431 | 1.023926 | 27 |
| block | 318 | 0.387085 | 40 |
| restore_early | 321 | 0.421509 | 46 |
| restore_mid | 427 | 0.980591 | 36 |
| restore_late | 318 | 0.382202 | 42 |
| reverse_early | 429 | 0.979736 | 28 |
| reverse_mid | 329 | 0.430908 | 41 |
| reverse_late | 434 | 1.024170 | 26 |
| random | 312 | 0.376465 | 45 |

development randomized control: 205925/222720 row/layer doses within max(1% target,1e-9); maximum relative error 0.41340501; 16795 unmatched rows retained and listed in phase summary.

confirmation randomized control: 411110/445440 row/layer doses within max(1% target,1e-9); maximum relative error 0.52331576; 34330 unmatched rows retained and listed in phase summary.

Dose is the actual BF16 **SDPA-output difference**, flattened across heads at each H row, before the unchanged output projection. It is not a norm match on projected residual states or logits. Targets are measured on T, per unit/layer/row, without gold answers. A seeded feature permutation supplies an unrelated direction from the current allowed-vs-blocked difference; zero directions use seeded Gaussian fallback. The closest BF16 realization is searched with fixed bracketing/bisection and all mismatches retained. The control matches T across all24 H layers; it does not separately dose-match each eight-layer partial restore. Thus T-vs-random supports whole-channel directional specificity, not independent semantic specificity of each interval.

`counterfactual_T_writeback_norms` in non-T records means a locally prospective allowed-G-vs-Z_G row difference on that condition's current trajectory; only T observe records are actual T targets. Random `dose.target` explicitly holds actual T targets.

Full block reproduces Z; full restore reproduces T. These are channel-construction and consistency identities, not independent path-discovery evidence. Partial restores, reverse blocks, factorial semantic signs and held-out confirmation supply scientific evidence. The experiment localizes a controlled map→later attention influence in K2, without identifying particular heads, exact S→map entry layers, a unique symbolic algorithm, or whether G composes versus relays an answer-relevant computation.

Actual confirmation dose diagnostic: 92.293% within the frozen1%/absolute tolerance; 816/445440 rows exceed10% relative error; sum absolute norm errors / sum target norms=0.02095%. Full quantiles and layer profiles are in RANDOM_DOSE_AUDIT.json. All rows remain included. The control approximates matched perturbation size; it is not an exact-dose null. This limits directional-specificity interpretation, especially individual low-amplitude rows.
