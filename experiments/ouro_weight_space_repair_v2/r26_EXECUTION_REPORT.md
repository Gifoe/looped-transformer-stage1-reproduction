# R26 finite original-path prefix acceptance

Post-run report. This is a numerical acceptance diagnostic, not an ARC repair candidate or an accuracy experiment.

The sole authorized job completed all five contexts and exactly 30 original uncached forwards. Supervisor PID 1795259 and worker PID 1795279 exited; supervisor status is complete, exit code 0, without timeout or overrun. The frozen analyzer ran once after complete output and parameter checks, and passed its input, vector, source, accounting and numerical integrity requirements. All named-parameter hashes agree before and after execution. Root independent review passed all 126 comparisons and 252 state/head domains, the full raw-vector/container/input hashes, all 15 ledgers, and 269 parameter hashes including agreement with R25.

## Result

All original-input repeats were bit exact. All protected same-shape suffix comparisons were bit exact at every requested native depth, for both normalized BF16 states and FP32 full-vocabulary head output. Across true prefix truncations, six of the 42 length-comparison pairs were nonexact (126 total pairs across all three comparison families): N0 C1 and C2 each at one of four depths, and key C1 at all four depths. State and head nonexactness occurred in the same six pairs. Q, N1 and N2 passed all tested comparisons; key C2 also passed.

The preregistered aggregate category is `length_bitwise_sensitive`. The tested same-shape future-content invariance passed, while the tested cross-length bitwise contract did not. This does not establish semantic future leakage, identify a kernel cause, or prove universal causality. It does not revive R25 or demonstrate any ARC improvement. No tolerance, backend change, extra probe, retry, accuracy, prediction or candidate ranking was introduced.

The largest recorded state/head absolute differences were respectively 0.0625/0.125 for N0 C1/C2 and 0.1875/0.1640625 for key C1. All comparisons use a fixed single-position [1,1,2048] BF16 state and [1,1,49152] FP32 head. Saved-vector exactness, byte mismatch counts and maximum differences were independently reconstructed by the frozen analyzer; CPU and GPU float64 RMS reductions are reported separately, without an equality tolerance.

## Reproducibility and cost

Definition commit: fa0bf55b46bb292394a5976f99c7d1e5f84bf7c8. Protocol SHA256: 03ef1e4369ef74b04c75b65bd30f8b2528603e1d21dec0828184d26433b1dcbb. The raw BF16 bit patterns and FP32 full-vocabulary vectors are losslessly retained in `results/r26_VECTORS.npz` (9668374 bytes, SHA256 2a2b3d86dbe91d4f03a9388cb86a81f1a1f893d09cd212f786519198d40701ec). Every comparison references content-addressed vectors.

Actual lifecycle cost was 55.08359357668087 seconds, including CPU revalidation, model load, full parameter hashing, forward/head computation, saved evidence and exit. Fifteen ledgers now sum to 18874.135196831077 / 21600 seconds, leaving 2725.8648031689227 seconds. The 14 prior ledger bytes and all frozen files remain unchanged. GB10 allocator measurements describe this bounded diagnostic, not deployment throughput.

No heldout split, test, other dataset, new method or follow-up model job was accessed. Upstream source and private tokenizer/transport files remain excluded from publication. Internal call evidence is source-bound to the original `_run` and actual loaded module/configuration, not a claimed interception of each SDPA invocation.

## Post-run distribution diagnostic

A separate CPU-only full-vocabulary distribution diagnostic found the truncation differences were not confined to a common logit offset: N0 native4 C1 total variation was about 0.00801 and C2 about 0.001914; key cut13 native4 was about 0.017677. These descriptive post-hoc values do not change the frozen acceptance result or identify a quantization/SDPA kernel cause, and do not quantify any prior accuracy impact. They use no argmax, target or label. See `results/r26_ROOT_LENGTH_DISTRIBUTION_DIAGNOSTIC.json` for the exact computations.
