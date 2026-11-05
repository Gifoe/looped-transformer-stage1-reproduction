# R31 shared-prefix binary capacity diagnostic

The one real CPU job completed all 24 questions and passed the frozen certificate analyzer plus final clean-supervisor join. It found **13 certified reachable, 11 certified unreachable for strict margin, and 0 undetermined** questions. All **10 R6 errors** are certified unreachable in this fixed binary family. The one R6-only gain is also certified unreachable. This closes the predeclared capacity challenge for a selector that can only choose between the already-observed Fixed and R30 C distributions at each shared Q prefix.

These are retrospective oracle capacity labels, not deployed predictions or an accuracy improvement. Gold defines the oracle constraints. The 13 reachable cases do not demonstrate an unlabeled selector, and they cannot be promoted into a method. Strict-margin unreachable means no assignment achieves every correct-versus-wrong score margin above the predeclared **1e-8**, after the conservative rounding enclosure. It does not by itself exclude a correct first-index argmax tie. The result does not close every source-value intervention, internal nonlinear mechanism or possible future distribution family.

Root's independent verifier rebuilt all 24 coefficient matrices and conservative bounds directly from the original LP values and prefix graph, without importing the oracle math/input/solver modules. It verified all 192 endpoints and all witness/simplex certificates, actual solver identities and unchanged 18 ledgers. Its retained receipt is results/r31_ROOT_CERTIFICATE_VERIFICATION.json. In these actual certificates all eleven unreachable upper bounds are strictly negative, so the first-index argmax tie rule cannot rescue any of those eleven. The thirteen reachable cases are exactly the A1-correct set and there are no cross-question shared keys. Therefore the oracle maximum correct count of this particular Fixed/C binary-routing family on these same 24 questions is exactly 13. This finite oracle bound is not a method gain or a claim about other distributions/interventions.

## Fixed family and complete input admission

All original 24 questions, 96 candidates and 548 position uses were kept. The 421 actual Q-prefix task keys define shared binary decisions; a key is shared by every candidate position that consumed that same model input. No key is shared across questions in this cohort, so separate question witnesses are jointly compatible. Candidate indices and target-token identity were not used to create independent routing keys.

A binary zero chooses the existing Fixed fullV distribution and a one chooses existing C. The oracle uses their already-saved FP32 target log probabilities, original character denominators and the exact binary value of each stored native late term. It does not create fullV logits from candidate scores, sweep coefficients/signs/clocks, load states or run a model. Only columns with exactly zero correction for every candidate are removed and fixed to zero. Input admission checked all 548 joins, finite nonpositive exact-FP32 LP values, original token/character counts and all 192 zero/one endpoint score enclosures. Maximum continuation length was 15.

## Exact certificates

Coefficients and error bounds use binary-rational Fractions. For each candidate, S is the sum of max(abs(Fixed LP),abs(C LP)) over positions, u=2^-24 and gamma=(n-1)u/(1-(n-1)u). The bound is `(gamma*S+n*2^-149)/chars + 16*2^-52*(S/chars+abs(stored_late)+1)`, evaluated entirely with Fractions. The admitted domain additionally requires `n*u<1/2` and `S+gamma*S+n*2^-149<=FP32max`, preventing the excluded overflow/long-reduction boundary cases. No actual candidate was dropped by these guards. Python/NumPy reductions are not treated as exact replicas of GPU FP32 reduction.

MILP proposes a strictly binary witness; the separate analyzer reconstructs candidate scores directly and requires every gold-wrong lower margin, with both candidate bounds subtracted, to exceed tau. Near-integer solver outputs are not silently rounded. LP proposes nonnegative simplex weights; after the predeclared tiny-negative clipping rule and exact rational normalization, the cube support-function upper bound includes both positive rounding bounds. Only an exact upper bound at or below tau certifies strict-margin unreachability. Solver infeasible/optimal/status/dual-bound strings are not certificates. All raw LP/MILP OptimizeResults and all exact coefficient/witness/dual records are retained.

The analyzer's output retains its original `complete_certified_pending_clean_supervisor` status. The separate `final_result()` join verified the clean final supervisor, output hashes, total CPU wall <=180 seconds and all 18 unchanged GPU ledgers before the counts in this report were released. The joined receipt is `results/r31_ORACLE_FINAL_JOIN.json`; the pending file alone is not a final-validity claim.

## Backgrounds

| Already-released background | Reachable | Strict-margin unreachable | Undetermined |
| --- | ---: | ---: | ---: |
| R6 correct | 13 | 1 | 0 |
| R6 error | 0 | 10 | 0 |
| R6-only gain | 0 | 1 | 0 |
| Other than R6-only gain | 13 | 10 | 0 |

These background rows overlap; only the three capacity classes partition all 24 questions. The R6-only gain remains `arc_challenge/train/LEAP_2001_4_10241`. The result gives no capacity basis for continuing a local binary Fixed/C selector on this pilot, and no new GPU execution is authorized.

## Actual execution and resources

Definition commit: `800e889ec702dc6fe03d1abdbd0b8a8516bb576f`. Protocol SHA256: `e5c02bdceb5680c80bb6988a408b6eedfe05fef983e745020f282b40e41cfc31`.

Supervisor PID 43896, worker PID 43428, analyzer PID 6516. Start Unix 1793846368.5639622, finish 1793846369.5721717. Worker and analyzer exited 0; no timeout or overrun. Total supervised CPU-job wall was **1.014999999984866 seconds**, including proposal worker, certificate analyzer and exits. Worker pre-final-receipt elapsed was 0.6090000000258442 seconds. The 180/175/179-second hard/TERM/KILL limits and per-MILP/per-LP three-second, one-thread limits were unchanged. CUDA was hidden. No retry or partial-data conclusion occurred.

Actual identity matched the frozen environment: Python 3.10.0, NumPy 2.2.6, SciPy 1.15.3, HiGHS 1.8.0, `D:\python-3.10\python.exe`, Windows-10-10.0.26100-SP0. Model loaded: false. New GPU-worker seconds: **0**. The 18 existing ledgers remain **20755.46525352029 / 21600**, leaving **844.5347464797087 seconds**; no ledger was reset or appended for CPU work.

## Records and scope

Raw proposals/certificates: `results/r31_ORACLE_RAW.json`, 360,846 bytes, SHA256 `f01a8b3e8d62b80cd8cba15c2e6e6e9e231a1b0573f0d3acab0490580f192d52`. The complete/analysis/final-join files, progress, supervisor and worker/analyzer logs are preserved. The definition, all nine terminal files and final join were synchronized to the server and matched by byte length/SHA; server verification ran only the read-only final validity join, not a second optimizer job.

The only labels were the same 24 already released by R30. No dataset gold file, heldout/confirmation/validation/test item, model weight or new inference was accessed by this oracle. R30 remains a completed negative method result. This diagnostic does not achieve the original research accuracy objective or replace the full 576+517+299+1172 evaluation required for a future method.
