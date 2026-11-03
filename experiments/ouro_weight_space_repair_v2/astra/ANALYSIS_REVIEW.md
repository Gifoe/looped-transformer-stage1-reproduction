# Independent V2 analysis review

## Final follow-up

Re-reviewed corrected source SHA256 `3fd96d50bb26b537a68290e8455e3e2dbe76aacd1bee6bb0f1d17407a7ac515e`. All three required findings below are addressed: exact stage sizes/candidate sets and per-row execution/data bindings are checked; changed_answer_cases now includes every changed prediction with a correctness_changed flag; native wide and long rows distinguish primary K4 from secondary depths. No remaining blocker was identified. The original-reference uniqueness and zero-denominator points remain small bounded hardening recommendations, not reasons to alter or delay the frozen experiment.

CPU synthetic checks were rerun against the corrected source and saved in `astra/SYNTHETIC_STATS_CHECK.json`; the reproducible test is `astra/check_analysis_synthetic.py`. This final run additionally checks the explicit two-primary Holm example (.03,.04) -> (.06,.06). It extracts only analysis function definitions and opens no data/result files. The report below preserves the initial review findings for provenance.

Reviewed source: `scripts/analyze_v2.py`, SHA256 `d92a3019b2592c543456b2680c27490bf2e3c82cf6fc994db0419ac38c7bd8bc`. This review read code only and ran CPU synthetic examples. No server test content or held-out outcomes were read. The reported real edited-path gates are implementation checks, not accuracy findings.

## Statistical implementation

The three-category multinomial bootstrap is correct for the paired accuracy difference. Each question contributes d in {-1,0,+1}; iid sampling n questions with replacement induces a Multinomial(n; damage/n, unchanged/n, rescue/n) count distribution. Both methods therefore give exactly the same bootstrap distribution for mean(d). Combining both-correct and both-wrong into the zero category loses no information for this estimand. This equivalence does not extend to arbitrary other statistics such as answer-change rates; the script uses the categorical bootstrap only for accuracy differences.

`binomtest(rescue, rescue+damage, .5)` is the two-sided exact McNemar test, and p=1 for no discordant pairs is correct. The Holm implementation correctly sorts p-values, multiplies by the remaining family size, clips at one and takes the running maximum. The two prespecified primary contrasts are native K4 edited-versus-original and Fixed LoopCD K4 edited-versus-corresponding-original. The all-six family additionally includes native K3/K5/K6 and Adaptive LoopCD. Separate explicitly named adjustments are defensible; reports must not choose whichever adjusted column gives a preferred significance conclusion. The bootstrap intervals are pointwise percentile intervals, not simultaneous intervals; the script labels that distinction correctly.

Synthetic checks passed: exhaustive n=3 enumeration of all 27 question-resampling outcomes versus multinomial probabilities; ID reordering; exact character normalization; rescue/damage sign; a simple exact-McNemar case; six-way Holm expected values and monotonicity; zero discordance; rejection of mismatched choice lengths. The test imported only the selected function ASTs, not main(), so no evaluation files were accessed.

## Required analysis corrections

1. **Bind complete stages and their rows to the execution locks.** Current receipt checks establish only `receipt.n == len(file)`, not 64/192/299/1172. Require the declared sample count per stage, unique expected IDs, exact candidate sets from SELECTION_LOCK and VALIDATION_LOCK, and the selected-only final candidate. Check each row's scope, candidate ID, implementation SHA, stage_data SHA and bundle SHA against its COMPLETE/SEARCH/MATERIALIZED/FINAL receipts. Otherwise a short but self-consistent COMPLETE file or unrelated extra stage can be reported as the prescribed experiment. The current 14/4/3/1 counts of candidate-stage files do not prove their identities or per-stage sample counts.
2. **Name and include changed-answer cases consistently.** `changed_answer_cases.csv` currently includes only C-to-W and W-to-C. It omits W-to-different-W, despite those being counted in the statistics. Either include every changed prediction and add `correctness_changed`, or rename the artifact to `correctness_flip_cases.csv`. All transition rows are already available in paired_transitions, so this is an extraction/name mismatch rather than a missing scoring calculation.
3. **Do not label all native depths as primary.** The wide native row currently gets role=primary while containing K3/K4/K5/K6. Only K4 is a primary contrast. Use explicit per-depth roles, or label the row `K4 primary; K3/K5/K6 secondary`. Fixed LoopCD K4 is primary; Adaptive LoopCD K4 is secondary. The actual Holm primary family is correct.

## Pairing and other checks

`align()` correctly reorders before rows to after IDs, rejects duplicate/mismatched ID sets in its arguments, and checks gold, prompt hash, character lengths and token counts. `validate()` recomputes all six supported normalized score arrays, argmax predictions, correctness and gold margins. It uses the declared character normalization and does not substitute token-length normalization. Fixed and Adaptive LoopCD are explicitly K4 only; the wide table correctly emits NA at unsupported K3/K5/K6 cells.

However, main() first maps an original-reference file into a dictionary and subsets it. Duplicate IDs in the original reference can be silently collapsed before align sees them. Validate uniqueness on each complete source file before mapping. Also verify that the original test/selection/validation ID sets exactly match the corresponding full staged manifests; only the screen intentionally subsets the 384-question training-statistics baseline. Binding to the existing staging/source hashes is preferable to assuming matching prompt/lengths uniquely identifies answer text.

The final test document map used for human-readable cases should verify 1172 unique IDs and matching IDs/gold/prompt/choice lengths before attaching question text. This is a pairing-integrity check, not permission to inspect test examples before the final lock. The script already requires PIPELINE_COMPLETE before opening these files.

For synthetic edge cases with zero original errors or zero original correct answers, the rescue/damage conditional rate is undefined; returning zero through max(1,denominator) can mislead. Prefer null plus its denominator. This does not affect ordinary nonempty-group estimates or McNemar/bootstraps.

## Expected output counts

- edit_search_results: 22 rows = 14 screen + 4 selection + 3 validation + 1 final.
- candidate_edit_audit: 15 rows = one tensor for 13 candidates plus two tensors for A08.
- paired_statistics: six test contrasts, with two rows carrying the primary-family Holm value.
- paired_transitions: 7032 rows = 1172 questions times six methods.
- final_comparison_long: 12 rows = two checkpoints times six supported readouts/depths.
- final_comparison wide: six rows = two checkpoints times three readout families; unsupported CD depths remain NA.
- generalization_by_split: 12 rows = the selected candidate's four splits times native K4, Fixed K4 and Adaptive K4. Search/selection/validation entries are descriptive and should not be sold as independent confirmatory evidence.
- changed-answer/correctness-flip counts are outcome-dependent; do not hardcode them.

## Verdict

The paired statistical formulas are correct. Fix the receipt/row completeness checks and the two artifact-label issues before publishing. The previously used official test remains a locked regression benchmark, not a blind test. Neither successful implementation gates nor this audit imply that the edited model improves accuracy.
