# R27 bounded functional source replay diagnostic

Post-run report; independent Root verification passed all raw vectors and content keys, finite values, five no-op pairs, 192 source pairs, 72 full-sequence first-three-loop comparisons, 40 identity state/head comparisons, all 33 forward records and clocks, 16 ledgers and 269 parameter hashes, including agreement with R26. Root independently recomputed the original margins and gates. This diagnostic does not evaluate ARC correction, fit a controller, identify a compact circuit or authorize deployment.

## Complete execution and integrity

The one authorized job completed all 33 forwards: one no-hook control, eight Phase A captures, and all 24 fixed Phase B replays. Supervisor PID1895579 and worker PID1895599 exited normally. There was no timeout, overrun, retry, profile or extra forward. The frozen analyzer ran once after clean completion and passed its complete schema, raw-vector, finite, cleanup, parameter and accounting checks.

All four query states and the native4 full-vocabulary head of the no-hook control were bit exact to the first capture. All 192 source K/V pairs were raw-bit exact across the two queries of each table/version. Every capture/replay followed the fixed 192-event projection clock. All 24 replays preserved the corresponding B baseline's first three loop states over the entire sequence; all eight K/V identity replays preserved four query states and full-vocabulary output. Source-row replay and unchanged other rows were checked in raw bytes. All 269 parameter hashes agree before/after and with R26, and all hooks were removed.

## Fixed functional result

Phase A passed the predefined eight-of-eight strict four-label subspace screen. This is not unconstrained full-vocabulary task accuracy. Stable full-vocabulary FP64 normalization gives four-label probability mass from 0.9477576649583839 to 0.9939709364536247 across the eight unpatched Phase A baselines (both A and B). Across all 32 evaluated conditions (eight baselines plus 24 replays), the worker range is 0.9477576649583839¨C0.9951950717223895. The maximum is the first table/first query V-shuffle replay, not a baseline. Root independently reports the all-32 range as 0.9477576649583828¨C0.9951950717223897; the tiny arithmetic difference is descriptive FP64 normalization roundoff. These ranges use different condition sets; none adds a gate or a full-vocabulary top1 claim.

The preregistered primary V screen passed. All four clean V margins crossed zero, all four transport differences were positive, and all four clean-versus-shuffled specificities were positive. Direct FP64 differences of raw FP32 logits were used, exactly equivalent to the specified log-probability differences without separately subtracting rounded normalizers.

| Instance | Original B margin | Clean V margin | V transport | V specificity |
|---|---:|---:|---:|---:|
| 0 | -4.5625 | 1.25 | 5.8125 | 5.8125 |
| 1 | -5.875 | 2.25 | 8.125 | 3.3125 |
| 2 | -4.4375 | 0.625 | 5.0625 | 4.9375 |
| 3 | -5.125 | 0.25 | 5.375 | 2.75 |

K was evaluated independently using the same fixed rule and did not pass: zero of four clean K margins crossed zero, and all four specificities were negative. K was not selected as an alternative success path.

The finite evidence concerns content correspondence transported through the fixed source V projections at loop4 of the specified synthetic task. Each table shares the same first-query donor across both queries, after exact source-query independence checks. It does not establish a V-only mechanism, a necessary individual head, a compact circuit, a native-fullV deployment coefficient, semantic correctness on ARC, or R6 benefit preservation. The cyclic permutation preserves the donor-vector multiset, but does not match all downstream activation distributions or eliminate every off-manifold explanation. No layers, loops, templates, labels, samples or thresholds were selected from these outcomes.

## Evidence and cost

Definition commit: 8d0ee1112fe7a6c092a85468e9891fd63f392ae6. Protocol SHA256: 9de2154c5d6ba207b6a2ad864410d2c555f4107ce34a4721ec5ba3c62cf1ea20. NPZ evidence contains raw BF16 bits and FP32 full-vocabulary arrays, content-addressed and deduplicated: 65938486 bytes, SHA256 c034a98c170d3d3bfcd89b818217fd1d8c77d1348ba71da6906912fc4418b6f6. B baseline full-sequence first-three-loop matrices are retained once; exact replay references share those bytes.

Actual lifecycle time was 56.29345832020044 seconds. Sixteen ledgers sum to 18930.428655151278 / 21600 seconds, leaving 2669.571344848722 seconds. CPU input revalidation, model loading, hashes, captures/replays, evidence IO, cleanup and exit are included. No heldout data, ARC outcome rows, other dataset or follow-up model job was accessed. Private tokenizer, transport, upstream source and model files remain excluded from public artifacts. Internal dispatch evidence is source-bound, not a claimed interception of SDPA.
