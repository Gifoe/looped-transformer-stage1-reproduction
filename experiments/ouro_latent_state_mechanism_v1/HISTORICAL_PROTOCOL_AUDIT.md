# Historical endpoint protocol audit

Scope: previously exposed ARC-Challenge test. This audit is CPU-only and reuses historical scoring records; no model forward or hidden-state capture was performed. It is not an independent evaluation.

## Provenance and completeness

- Original revision: ByteDance/Ouro-1.4B `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; historical receipt records checkpoint SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`. Historical BF16, SDPA and no-cache context comes from the earlier experiment receipt, not a new runtime check.
- Existing historical CSV SHA256 `9cd8af62b4465be1e6066f6ffd0cd965c1f56c1e03fed13baf5b59e919c80a5f`. Its 4,688 joint/cyclic × K1/K4 vectors were independently checked exactly against the original 1,172-row `r11_test_selected.jsonl`, SHA256 `93845afb6a449590e465dac8ed8430e22ee618e64559f19a8fec1ff805eff9da`, read from the verified two-part archive SHA256 `b172b56ccaad87ba49344fdd20b9a77b6293a0dd2b3e10fc9571d97ab260a4ac`. Source part lengths and hashes were checked.
- All 1,172 official test IDs are present exactly once per view and depth; 3-, 4-, and 5-choice sizes, gold labels against the original option labels, finite scores, exact first-index argmax and correctness were checked; stored margins match computed margins within 1e-6. No missing ID or silent intersection.
- Fixed 25-shot example file SHA256 `8cf4b70e7c07c67a589560d5fcbc1fa8feb85a0e7af3b348fa21f147271419da`. For every rotation, source question, display order and historical request hashes were recreated against the frozen examples. All historical continuation lengths were 1 token.

## Definitions and verification

Joint = the original order, all options in the prompt, a space plus candidate letter scored via full-vocabulary FP32 log-softmax/gather/sum. It is not normalized by option-text length. Cyclic pooled = arithmetic mean of the candidate log-scores of every cyclic rotation, after mapping back to original option indices. It includes the original order and is neither one permutation, a vote, nor an independent sample. Historical selection uses first argmax in original option order, including ties.

For every question and both depths, rotation 0 was confirmed to be original order and its score vector exactly equal to the joint vector. The stored pooled vector exactly matched the mean of all mapped rotation vectors. The direct mean over non-original rotations exactly matched `(n * pooled - joint) / (n - 1)` in the audited float64 calculations. This leave-original-out vector is still a pool of the same question and examples; it is a descriptive sensitivity comparison, not independent validation.

Only K1 and K4 are available for the full cohort. The 2-question A4 data are technical checks; K2/K3 are not imputed or used to localize change. Equal endpoints cannot exclude an intermediate reversal. The code `historical_endpoint_observation.py`, complete machine-readable question table and summary are under `results/HISTORICAL_ENDPOINT_OBSERVATION.jsonl` and `.json`.

## Rotation mapper example

At the historical base commit, `scripts/r11_inference.py` lines 24–30 forms `order[j] = (j + shift) % n` and sets `v[i] = vector[j]` for each `j, i in enumerate(order)` before storing `mapped[name]`. The source SHA256 is `0c975cc13edb770d8e2788ca5048bdfa4b4f4fc614937c7c1af4d2a99bc79892`. Its `infer_auxiliary` call is used by the historical R11 worker and returned rotations feed `joint_baselines`, where the mapped vectors are averaged.

For `Mercury_7175875`, shift 1 has `order=[1,2,3,0]`, so displayed indices 0,1,2,3 map to original indices 1,2,3,0 respectively. The archived K1 mapped vector is `[-2.6882240772247314, -2.1882240772247314, -0.3132239878177643, -2.4382240772247314]`. Applying the mapping in reverse yields the display-order vector `[-2.1882240772247314, -0.3132239878177643, -2.4382240772247314, -2.6882240772247314]`; the display-order vector here is reconstructed from the stored mapped vector, not claimed as independently archived.
