# Independent V2 pipeline source review

## Follow-up status

The corrected runner SHA256 `bc3eee7737670a66f2c5ab7aab3d3603782c45452d2d73407b529abb2530b242` resolves both required issues below: SEARCH_LOCK binds DATA_LOCK and all four training input file hashes; every evaluation row and COMPLETE binds a canonical hash of the full stage data; resume checks that hash, scope, candidate and lengths. Family quotas are now explicit in CANDIDATE_DESIGN and the protocol's design hash matches the actual file. Source parsing passed. No remaining launch-blocking defect was found in this follow-up. Gate/bundle receipt checks below remain bounded hardening recommendations. The original findings are preserved as audit history, not unresolved blockers.

Reviewed `scripts/run_edit_pipeline.py`, `weight_edits.py`, the scorer's scoring and depth-check routines, and the frozen-design description. This is a code-only review before edited-candidate feedback. No GPU execution and no selection/validation/test outcome reads were performed. No V1 file was modified.

## Required before launch

### 1. Bind complete input manifests to locks and resumed rows

At lines 117–123, SEARCH_LOCK binds code and candidate specifications but not DATA_LOCK or the three training manifest hashes. Lines 140–145 only check a subset relation, disjoint IDs and two lengths. Lines 177–181 accept previous rows based on ID, implementation, bundle, prompt hash and gold. A change to candidate text/request continuations, normalization lengths, or another field can therefore leave these checks passing while mixing old and new partial results. A stale or accidentally replaced screen subset with valid discovery IDs also passes.

Patch before any candidate evaluation:

1. Read DATA_LOCK and verify its discovery_expanded and discovery_screen cohort file hashes/counts, plus selection_sha256 and demonstrations_sha256. Verify 384/64/192 unique IDs and screen-subset/discovery-selection disjointness. Bind DATA_LOCK hash and all input file hashes in SEARCH_LOCK.
2. Give evaluate() a dataset_manifest_sha256 argument. Record it in every row and COMPLETE. Verify it for every resumed row, alongside `candidate_id == cid`, `scope == stage`, exact choice_char_lengths, method-key set, and expected ID/gold/prompt. This is stronger and simpler than attempting to reconstruct all input equivalences from a few fields.
3. Include the validation and test staged-manifest hash in their evaluation receipts. The staging routine already compares original and destination hashes; preserve that binding through COMPLETE and FINAL_LOCK/FINAL_STAGING receipt. Do not read test rows earlier to obtain it: a metadata-only file hash or a post-lock staging receipt is sufficient.

This is an integrity/resume issue, not evidence that the current data have changed. No dataset corruption is asserted.

### 2. Freeze the family-constrained promotion policy consistently

The code's `promote()` deliberately promotes two candidates per family from the screen, then one per family plus the best remaining candidate from selection. This differs from the method review's unqualified “top4” and “top3” and is not explicit in CANDIDATE_DESIGN. It can change who reaches validation. The script does freeze its actual policy in SEARCH_LOCK before evaluation, which prevents a post-feedback switch, but the authoritative design and report should name the same rule.

Add explicit screen/selection family-quota fields to the design and frozen configuration before running edited candidates; regenerate/update dependent design hashes and documentation coherently. Alternatively use the already-described overall top4/top3. Do not choose between these policies after inspecting edited results. Within each family and among the remaining candidates, native K4 correct count, mean gold margin, actual BF16 changed coordinates, and ID are implemented in the declared order. Final validation ranking has no family quota and selects exactly one edited candidate.

## Additional bounded fixes

- At lines 169–170, resumed EDITED_PATH_GATE verifies only implementation hash. Require status passed, exactly A01/B01, the current bundle hashes, training-question ID, exact own-logit/depth/restoration flags, and nonzero changed counts/scores. The surrounding materialized-weight lock reduces the risk, but the gate receipt should be independently bound to what it certifies.
- When loading an existing bundle at line 129, also compare `metadata.candidates_frozen_sha256` with the current frozen-file hash. Candidate-spec equality plus SEARCH_LOCK already provides substantial protection, but explicit provenance should agree.
- Resume currently fails closed on a truncated trailing JSONL record and on an interrupted partial bundle/copy. This does not silently contaminate results, but it is not transparent crash recovery. Document the fail-closed behavior. If automatic recovery is implemented, repair only the incomplete trailing line and preserve its bytes in an audit artifact; never drop complete records or recompute a candidate from changed factors.
- Original baseline rows are copied only after selected test evaluation, which satisfies the access ordering. Their compatibility check currently binds ID/prompt/gold but should also verify choice lengths and candidate count when pairing outputs. Preserve their baseline implementation/model provenance in the final report; V2's changed-weight trials are not baseline reruns.

## Paths verified by source inspection

- All 14 edited candidates are constructed before screening. Every subsequent stage calls load_edit_bundle, which reconstructs exact BF16 bits and verifies the original base tensor, full edited tensor, payload, factor and count hashes. There is no repeated low-rank GEMM between stages. The selected directory is a byte-preserving copy followed by a full bundle verification.
- applied_edits changes real physical projection parameters with copy_, covers all recurrent invocations, restores exact clones even on exceptions, and invalidates scorer state at both application and restoration. Embeddings/head/norm parameters are outside the allowed names.
- Each candidate's single K6 trajectory supplies native K1–K6 scores and both LoopCD readouts. own_logits_check independently applies the BF16 head to the same edited forward's saved normalized K1/K4 prediction states and requires zero discrepancy. No original-model states or logits enter this computation. The separate-depth gate uses the edited parameters. The gate covers one answer candidate each for A01/B01, not every candidate/rank/site, so reports must state that scope.
- Official validation rows are staged after VALIDATION_LOCK; test rows after FINAL_LOCK. The final selected edited checkpoint remains eligible even if inferior to baseline, and baseline is never a selection candidate. The final stage does not branch on test scores to change weights, depth or coefficients. K3/K5/K6 are recorded, while the selection key remains native K4.
- Existing score rows carry implementation and bundle-manifest hashes; rerunning completed stages performs no new candidate forwards. Lock equality makes incompatible code or candidate changes fail closed. Completion summaries are recomputed in manifest order, so deterministic resumed rankings are expected once complete input hashes are also bound.

## Verdict

Proceed after the input-hash/resume fix and promotion-policy alignment. The reviewed source implements actual edited-model evaluation and own-edited LoopCD; no additional V1 causal gate is appropriate. This review does not establish speed, real-model gate success, or empirical benefit.
