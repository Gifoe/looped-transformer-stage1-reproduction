# R30 source-clock pilot: complete negative result

The frozen source-clock C pilot completed all 24 questions, 96 candidates, 548 target-position uses and 389 shared-reference groups. Primary was correct on **13/24**, versus **14/24** for both matched and historical R6. Against either R6 there were **0 rescues, 1 damage, net -1**; the one R6-only gain was not retained (**0/1**). Both the original strong screen and the separate nonpass cost-review eligibility flag are **false**. No full-chain execution or additional GPU experiment follows from this result.

Primary predictions equal A1, M3, J2 guidance, G-matched, vocabulary-null, matched_scale and both leave-one-reference readouts on all 24 questions. Fixed differs on one wrong-to-wrong prediction despite the same 13/24 count. M2 has 14/24 and remains an ineligible control. Neither its count nor another control can replace the frozen primary.

The new correction was active at **236/421** unique Q prefixes and **306/548** target-position uses. Thus the implementation did not universally reduce to zero correction. Activity did not produce the required R6 rescue. This finite reused pilot does not statistically exclude a small full-ARC gain, and it does not establish the causal reason for the failure.

Root's complete all-24 descriptive inventory found that C increased the gold-minus-best-wrong normalized score margin relative to A1 on 9 questions, left 2 unchanged and decreased 13 (mean change -0.0027848812248611374). Against matched R6, the corresponding counts were 7/1/16 (mean -0.009605450439632039). Relative to R28 spatial A, margins increased on 12 and decreased on 12; primary predictions nevertheless agreed on all 24. These observations do not support attributing failure solely to a universally negligible signal. They are descriptive observations on the same already-unsealed pilot, not a new estimator, independent efficacy test or causal attribution. The full 24-by-four candidate inventory is retained in results/r30_ROOT_ALL24_MARGIN_INVENTORY.json.

## Intervention and immutable primitive

Each N1/N2 true-prefix CF forward captured each physical layer's own V at loop1, then replaced that layer's loop4 V at the same 143 fixed demo-answer positions. There was no spatial permutation or cross-prefix donor reuse. Q and N0 were not intervened on. The model, BF16 weights, uncached/no-explicit-mask SDPA path, single-position BF16 head shape, full-vocabulary FP32 normalization, original FP32 position buffers/one sum, original character denominator and native Q4-Q5 term remained fixed. The main fullV projector and all controls reused the frozen R28 math_v2 bytes. This intervention contains temporal scale/content/distribution changes; it is not a purified semantic prior.

Definition commit: `391763350c25921785d79ee9d338f33e53969f9e`. Protocol SHA256: `c2cb2bfbe0a81c27be9e4364fad4e92ab1d2e65e374c881c0d5cb05c8e55787a`. R28 A remains closed, and its spatial-CF task identities were not reused as clock-CF cache entries.

## Completed integrity and audit scope

Before any new CF feature, the first native-cache pass completed 3597 head evaluations. All eleven baseline families' 6028 target LP values, candidate sums, normalized scores, late terms and matched R6 agreed strictly with R28; all nine historically saved representative fullV heads were bit-exact. Old fullV outputs were not saved for every task, so all-task historical fullV bit parity is not claimed. The cache ABI covered 1588 native tasks and 6773 BF16 single-position state references with exact hashes/input/context/depth/primitive/read-position bindings.

The second native pass completed another 3597 heads. All **785 model-body forwards** completed: 778 CF plus one fresh Q, two capture-only references, two clock identities and two CF repeats. These body forwards produced 787 head evaluations. Total head work was **7194 cached-state heads + 787 model-body heads = 7981**. There was no warmup, separate profile, retry or extra model call.

All 2334 CF first-three-loop **last-position** comparisons matched the native cache. The two fixed representatives additionally passed full-sequence capture-only/native and CF first-three-loop checks, same-layer loop1 source-donor checks, identity and repeat checks. Full-sequence evidence is representative scope, not coverage of every intermediate token for all 778 CFs. The 269 parameter hashes matched before/after and R28, hook cleanup was clean, controls were matched, and no nonfinite/schema/accounting failure occurred.

The frozen official analyzer ran once after complete clean termination, performed the entire label-free numeric gate, and then unsealed the original 24 labels. Root's postrun CPU helper independently compared retained cross-file raw tensors: eight old native full states, two old native heads, six CF full first-three-loop states, 48 source-donor pairs, eight identity states/two heads, and eight repeat states/two heads/48 donors. It independently recomputed the representative fullV projection (maximum error **1.1102230246251565e-16**) and already-unsealed paired/retention/gate arithmetic. It reran the frozen numeric_integrity function; it is not a second independent implementation of that entire function. No local or additional model execution was needed.

## Actual resources

Supervisor/worker PIDs were 2474130/2474150; both exited. Supervisor start was Unix 1793843729.216908 and finish was 1793844290.5965023. Exit code was 0, without timeout or overrun. The fixed hard/TERM/KILL limits remained 700/685/699 seconds, with the five-second reserve.

Actual eighteenth-ledger debit: **561.3795881131664 seconds**. All eighteen ledgers, each used_seconds counted once, sum to **20755.46525352029 / 21600**, leaving **844.5347464797087 seconds**. Component timings are subsets of this actual lifecycle and are not added to it again.

| Diagnostic component | Seconds |
| --- | ---: |
| All model-body forwards | 447.573620 |
| Cached-state head evaluations | 12.224022 |
| Model-body head evaluations | 1.740279 |
| Native cache decode/transfer | 10.527358 |
| Projection/scoring/parity reduction | 41.349544 |
| Model load | 15.439455 |
| Before/after parameter and identity hashing | 14.910804 |
| Categorized checks | 1.966306 |
| Evidence IO capture/export | 0.412167 |

Worker pre-final-receipt elapsed was 559.249999842 seconds, measured components summed to 550.638613723, and uncategorized worker residual was 8.611386118. Final receipt/exit/supervisor overhead is included in the actual ledger. Timings describe this synchronized diagnostic pipeline with hooks and checks. They are not uninstrumented deployment speed. GB10 has unified memory; peak Torch CUDA allocated bytes were 3029316096, which is not a complete device/system memory accounting.

## Records and limits

All sixteen terminal data/log files were downloaded and matched server byte lengths and SHA256. The lossless raw vector container is **75,763,006 bytes**, SHA256 `c26f00d207362687a3f1936e453e08c7b4bc4ce8f7e0b56ec13a6b724a30350a`. Scores SHA256 is `5f35d91b38c21648879019900d8f613274e505c0240b7b5063fb5a8c7737c677` and analysis SHA256 is `960a490cb46b4d5173bd2b17b821f38c3e5f938a93f97976f1f78dc2610922b2`. Root postrun verification is `results/r30_ROOT_POSTRUN_VERIFICATION.json`.

No confirmation517, validation299, test1172, external benchmark, training or weight update was opened. R29 canonical execution was not mixed into this job. A subsequent full 576+517+299+1172 study is not authorized; neither strong nor cost eligibility passed. The original research accuracy objective remains unachieved.
