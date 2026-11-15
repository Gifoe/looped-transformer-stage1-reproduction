# Scientific review and the selected missing edge

2026-11-14. Astra directs scientific design; current Sol owns all numerical analysis, implementation and execution. No new raw-data computation or GPU experiment was performed by Astra for this review. This is an experiment-design review using the ccf-experiment-designer skill, not a new result report.

## Current judgement

The evidence supports a conditional distributed update, several local readout pathways, and important failures of simpler accounts. It does not yet explain the complete natural K1-to-K2 improvement. The most useful missing edge to test now is **the validated L10–L16 conditional attention-reference update -> the actual L18 residual/attention state -> decision**. C1 freezes this test in ASTRA_ORDER_C1_UPSTREAM_TO_L18.md. It is neither a repeat of L-only localization nor a repetition of the old natural-K1-state L18 transplant.

Calling E3's L-only experiment pending is now incorrect. The completed1030/299 report and root's independent arithmetic audit replace that status, while older snapshots remain unchanged. Calling its d=N−P a natural K2−K1 update is also incorrect: P rolls back Q-region states at L9 to same-question K1 states. Subsequent d values include feedback and context changes. We must identify which actual changes reach the output before attributing them to answer refinement, knowledge retrieval or new reasoning.

## Existing evidence and limits

| Evidence | What it supports | What it does not establish |
|---|---|---|
| E1 fixed-K1 receiver SDPA interaction removal and suffix decomposition | A measurable K/V interaction component affects this mixed receiver; suffix non-additivity partly offsets it in rescue groups | Necessity on natural K2, an MLP-only source, or a complete recurrent computation |
| E2 frozen raw/unit probes and state interventions | Some answer-associated information is linearly available at recorded R sites; fixed probe transfer improves from K1 to K2 | Native causal use, absence of information after a failed probe, or an exhaustive full-sequence boundary representation |
| E2 rank-4 versus offered-candidate projection | The original span has a major extra E-axis and unequal doses; its interpretation requires candidate/scale controls | A uniquely identified answer subspace, or greater information importance inferred from larger complement perturbations |
| E3 L10–L16 N−P reference updates | Bidirectional conditional restoration/removal and specified norm/unrelated contrasts pass in both previously exposed Train1030 and Validation299 | A unique writer, natural K2−K1 dynamics, or the downstream mediator of this particular L-only intervention |
| E4 L18 natural-K1-input rollback and actual-attention four corners | Certain natural corrections can be restored through the local attention branch; residual/suffix routes and contrary cases persist | Universal attention mediation, a mediated percentage, or the source of the naturally different L18 input |
| E5 matched later-depth intervention | Effects and failures change across depth; late mean restoration can be negative and K4 can damage correctness | Monotonic refinement or decreased sensitivity at identical dose; natural perturbation magnitudes also change |
| E6A/B and H0/H1/H2 | Frozen output/latent policies and diagnostics failed their overall incremental gates; matched weak-comparator improvements have limited scope | No hidden information exists, or further feature/accuracy search is justified without a causal question |

E3 natural-rescue means passed while many individual questions did not satisfy all six directional contrasts. Exact BF16 arithmetic gives65/306 Train natural-no-tie cases with at least one nonpositive contrast, and14/100 Validation. AKDE&ED_2012_4_21 and Mercury_SC_415540 have exactly zero specificity contrasts that production floating subtraction reports slightly positive. Such cases prohibit interpreting a positive mean as a universal path.

E4 confirmation likewise matters: strict KV necessity did not reproduce in128; the K2 attention-restoration mean interval crosses zero, although two strict individual recoveries remain. ACTAAP_2007_7_25 retains damage after attention restoration; NCEOGA_2013_8_4 can be damaged by restoration itself. The C1 design must retain these categories instead of aiming merely to recreate a favorable mean.

## Why the proposed bridge is identifiable at its stated scope

U only changes the final R token's attention additions at L10–L16. Under the audited causal/no-cache computation, non-R context stays natural. At L18 U has an actual xU and an actual attention increment aU. Restoring xN before attention would also alter Q/K/V and confound the two routes. C1 instead computes U attention first, then crosses the actual residual-add branches, followed by the complete real MLP/suffix.

Joint xN+aN restoration necessarily returns the natural suffix when non-R context is identical. It is an implementation identity, not a discovery. The informative measurements are the two single-branch conditional effects, their reciprocal losses, controls, interaction, and same-question decision reversals. P has a different non-R context; the natural identity guarantee is not applied to P or P+L.

Source inspection identified a concrete hook hazard: replacing a norm's input tensor without replacing the already-saved MLP skip implements the wrong intervention. C1 requires explicit alias/skip verification and an independent injection implementation. A patch that merely changes final scores cannot pass this gate.

## Remaining chain, regardless of C1 result

Actual recurrence is K1 L24 output -> shared final RMSNorm on the whole sequence -> K2 L1 input. There is no embedding reset and no old-loop KV reuse in the no-cache path. Existing probes usually see only R. Even a completely supported C1 route leaves how the full recurrent state generates Q9 and the subsequent conditional updates unresolved.

The next decision must use C1's actual results and reuse E2's frozen models/subspace controls where applicable. Do not refit a new probe and call improved decoding the missing causal edge. Do not convert a failure of direction summaries into an assertion of information absence. No new accuracy controller, E6C, automatic E7 policy branch or broad layer/head scan is authorized by this review.

## Directly inspected sources and review scope

The new user command was read in full from C:/Users/gifoe/.codex/attachments/fc1f35ae-5a42-4b05-96ec-26ae7b64d4b7/Pasted text.txt. The following are direct document/code reads, not claims of newly replicated computation:

- D:/ouro-e1e2-publish-20261113/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E1_E2_A6000/E1_E2_RESULTS_AND_SCOPE.md; source/E2/probe_test_once/E2_FROZEN_TEST_REPORT.md; local_reviews/handoff/E2_V3_INDEPENDENT_AUDIT.md. The latter is an earlier fit-stage audit with a then-pending Test evaluator; the later frozen report is not erased by that historical status.
- D:/ouro-e4-e5-20261112/publication_staging/e3_writer_increment_20261113/E3_ASTRA_REVIEW.md, E3_TARGET_DEFINITION.md, E3_PATH_PATCHING.md, orders/ORDER_E3_FIRST_READ_VS_LATER_WRITES_CONDITIONAL_v1.md, code/e3_conditional_factor_hook.py and code/audit_l_val.py. These establish the actual operator and audit scope.
- The same E3 package's results/conditional_l_holdout/INDEPENDENT_FRACTION_COUNTEREXAMPLES.json, including both named exact-zero records. This is direct examination of reported question-level counterexamples, not a fresh all-row computation.
- D:/ouro-k1-k2-mechanism-20261114/PRIOR_CONDITIONAL_L_ORDER.md, PRIOR_L_INDEPENDENT_ARITHMETIC_AUDIT.json and RECURRENT_BOUNDARY_SOURCE_AUDIT.md. Root's1329-row PASS covers derived endpoint arithmetic and coverage; it explicitly does not duplicate the entire tensor/source/bootstrap audit.
- D:/ouro-e4-e5-20261112/E4_MEDIATION_ANALYSIS.md and E4_FAILURE_CASES.md, including fixed128 and late-depth failures. Historical final reports' peer-PENDING comments are superseded only by the newer E3 evidence, not silently edited out of history.
- D:/ouro-e6a-20261113/publication_snapshot/E6A/E6A_BASELINE_COMPARISON.md; E6B/E6B_ASTRA_SCIENTIFIC_REVIEW.md, E6B_OUTPUT_VS_LATENT_ABLATION.md and E6B_COUNTEREXAMPLES.md; next_step_train_diagnosis/TRAIN_DEV_FAILURE_DIAGNOSIS.md. The E6B review explicitly states it is Sol-authored despite its filename; Astra does not claim its authorship or computations.
- D:/ouro-e7-20261113/publication_snapshot/H2_candidate_direction/analysis/H2_SCIENTIFIC_ASSESSMENT.md supersedes the earlier H0/H1 report's H2-NOT-RUN status. It records the real per-layer capture, failed prospective gate and no G2/Test expansion.

This review is not a claim that Astra personally inspected every raw tensor, every E1–E6 implementation file or every question row. Those scopes remain owned by the existing independent audits and Sol's source/coverage reconciliation. No conclusion here relies on an uninspected file being complete merely because its name exists. The frozen C1 source manifest must bind the exact files actually reused and expose any remaining prerequisite discrepancy before GPU execution.
