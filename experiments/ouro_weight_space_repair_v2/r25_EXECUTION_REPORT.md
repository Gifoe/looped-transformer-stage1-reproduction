# R25: complete 24-question pilot; both continuation rules failed

The sole supervised pilot completed all 24 original development questions in 177.252 GPU-worker seconds. It exited normally without a timeout or overrun, and all 269 parameter hashes matched before and after execution. The method, controls, cohort, prompts, candidate order, character denominators and late term remained fixed.

The original strong screen failed. The separately declared `cost_review_eligible` rule also failed: the primary method had a negative net change against R6 and lost the sole old R6-unique correction. This closes the exact R25 pilot without a budget request, control promotion or further model job. It does not statistically exclude every small full-ARC effect or every memory method.

| Method | Correct / 24 |
|---|---:|
| Primary | 13 |
| Original M3 | 13 |
| Original M2 / R6 | 14 / 14 |
| Q-only / reference-only | 13 / 13 |
| Frequency / raw occurrence shuffle | 13 / 13 |
| Frequency-bijection matched null | 7 |
| Memory-direction-only | 1 |
| Each of three leaveouts | 13 |
| Fixed / A1 | 13 / 13 |

Against R6, the primary rescued 0 questions and damaged 1: net -1, exact two-sided McNemar p = 1. The one old R6-only-versus-Fixed success was not retained. All three leaveout stability requirements failed. The primary did not beat M3, Q-only, reference-only or raw occurrence shuffle.

Equal accuracy does not imply identical predictions. Primary and M3 differed on one question, but both answers were wrong. The R6-relative correctness loss was already present in M3; this memory increment failed to recover it, rather than establishing that memory itself caused that loss. Primary predictions were identical to Fixed, Q-only and each leaveout on all 24 questions. Its one prediction difference from raw shuffle also produced no net correctness gain. The poor memory-direction-only and matched-null controls do not by themselves establish useful semantic content; both remain ineligible.

Before model loading, a CPU audit checked all 24 prompts and every candidate against the fixed shared-demo key input, all 143 value offsets and predecessors, and original target signatures. Selected-position versus same-input full-position native-4 BF16 states and full-vocabulary heads matched exactly. Fresh Q4 matched cached Q4; N0, N1 and N2 repeated native-4 heads also matched exactly. These are four Q/reference readout-context checks, plus the separate key selected/full audit.

Independent NumPy calculations for the first query in each of the four contexts reproduced all 143 stored distances within 1e-12 and selected the same neighbor. Original native/reference sums, baseline vectors, M2/M3 alphas and final scores were exact. The HF call evidence binds immutable original `_run` source, entry arguments and backend; it is not a call-interception receipt or an independent causal-mask proof. No hooks, masks or KV paths were changed.

The frozen analyzer completed its entire label-free integrity pass over all 24 questions before loading gold or computing predictions. It checked schema, inputs, parity, parameters, neighbor distances and gaps, control norms and ledgers, then saved `LABEL_FREE_INTEGRITY`. There were no unmatched positions; maximum norm error was 3.331e-16. The primary correction was zero at 183 of 548 positions. Query and reference retrieved the same value in 742 of 1,644 comparisons (45.13%). Raw occurrence shuffle collapsed a nonzero primary correction to zero at two positions; the frequency-stratified value-ID bijection preserved zero qualification as designed.

Root independently verified local provenance, raw-score arithmetic, null maps, all 548 sparse directions, predictions, paired statistics and resource flags. That review did not run a local model or independently regenerate full-vocabulary key/query distributions. Its helper and receipt are included in the archive.

A separate post-run CPU content diagnostic re-tokenized all 96 candidates and reconstructed ideal sparse corrections at their actual target IDs. Only 82 of 548 positions had a nonzero direct target term: 70 positive and 12 negative. Forty-two candidates had no direct target term. Gold candidates had 21 supported positions out of 148; incorrect candidates had 61 out of 400. These are retrospective descriptions, not semantic efficacy evidence, a new estimator or a basis for selecting parameters. The reported partition residual also contains FP32 casting, log-softmax and summation error; it is not an exact isolated partition effect. The private tokenizer used for this independent audit is excluded from public files and archives.

| Measured component | Seconds |
|---|---:|
| Model load | 16.102 |
| Key build / key audit | 1.082 / 0.578 |
| Q head | 0.770 |
| Reference forward / head | 114.265 / 0.790 |
| Retrieval and update | 10.485 |
| Explicit checks | 2.263 |
| Total supervised worker | 177.252 |

The component timers do not exhaust lifecycle costs. CPU input audit, parameter hashing, remaining normalization, file IO and shutdown are included in the total. Cached-main pilot throughput is not independent deployment speed, and the GB10 allocator peak is not total unified-memory usage.

All 14 ledgers, counted once by `used_seconds`, total 18,819.051603254396 of 21,600 seconds, leaving 2,780.9483967456035 seconds. A complete evaluation still requires 576 development, 517 confirmation, 299 validation and 1,172 ARC questions. Its existing lower planning estimate of about 10,815 seconds, before extra retrieval audits, exceeds the remaining cap. No confirmation subset was substituted.

The development cohort has been repeatedly reused. R24's 62/216 token-content admission represented a limited content opportunity, not achieved rescues or proof that retrieval could identify correct answers. No confirmation, validation or test question records, other datasets or external benchmarks were accessed. The wider research goal remains unachieved.

Definition commit: `27077b2bc931dacb60b8d8c7c712707df14aa3e9`.
Protocol SHA-256: `a2ba1d236eb1ace9d60af16e2b574dfb018c494a7496c87199f8e41f5c9d8641`.
