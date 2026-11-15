# AB1: actual recurrent boundary Q×R inheritance and Q9 path crossing

Status: scientific design frozen by Astra; no model run performed or server chosen by this document. Terra owns implementation, mutual review, source reconciliation and execution after root resolves the authorized server. No routine Astra gate is required. Preserve running jobs. This is one bounded causal experiment, not a layer/head/token search, a new probe fit or E6-C.

## Why this experiment, rather than repeating C1

The starting user release is commit `1693cd0490282776e2a82c57397c74ad24fbc1ec`. Preserve that immutable starting reference and bind the actual reused files/commits in Terra's source manifest; newer C1 evidence is an explicit addition, not silently attributed to that commit.

C1 is complete, despite stale ACTIVE_RESEARCH_STATE/EXECUTION_STATE wording. Its fresh 5090 Train1030/Validation299 native-rescue counts are342/102. Both actual L18 X and A routes pass their frozen bidirectional/control family bounds. X restoration means1.52985/1.48182 and A restoration .20108/.26838 concern the specified Q9-conditioned N−P L10–16 subtraction. They do not reveal how the real full-sequence K1 boundary generates Q9. Whole X+A restoration is structural. The eight-case FP32 math/deterministic run preserves K2 endpoint predictions but changes weak effects; it changes backend/settings as well as dtype. Do not call it isolated precision or population validation. Repeating C1 is not authorized.

E2's original 16 frozen heads/scalers have now been recovered, with exact expected hashes; the older missing-artifact report is superseded by E2_PARAMETER_RECOVERY_STATUS.md. Full-sequence boundary states have not been recovered. E1/E3 local effects, E4/E5 subset mediation and failures, and E6/H2 failed incremental utility do not close the boundary-generation edge. In particular a deterministic recurrent pass need not add information in the information-theoretic sense to create a representation newly usable by a finite decoder or native computation.

## Fixed question and rival explanations

Does K2's benefit depend primarily on inheriting contextualized current-input states from K1, on inheriting the readout state, or on their interaction? If contextualized Q inheritance matters, does its effect reach the output through the Q states actually generated at L9, rather than only through R changes acquired before L9?

These tests can distinguish causal dependence on inherited state and conditional computation. They cannot establish that K1 lacks information, isolate its original semantic source, or distinguish all possible forms of integration from reorganization.

## Exact states, sites and common receiver

Use unchanged pinned model/revision/weights, BF16, SDPA, eval, no cache, original25-shot Joint-Option letter scoring and complete offered candidates. Re-establish same-site K1/K2; never merge A6000/4090/5090 groups. Model forward inputs contain no gold. Full-vocabulary production scoring and first-offered ties remain unchanged.

Capture E: actual full-sequence input to physical L1/index0 at current_ut=0, after embedding lookup and before the first decoder layer. Capture B: actual shared final RMSNorm output after K1 L24, and verify B is bitwise the full-sequence L1 input at current_ut=1. The intervention is **after** that final RMSNorm, at K2 L1 entry. Do not normalize B again, add embeddings again, substitute K1 L24 pre-norm output, or carry old-loop KV.

Use verified token-ID/offset regions P=fixed demonstration prefix, Q=all current-input tokens excluding R, R=last token. Q includes options/format. P/Q/R must be disjoint and complete; prefix/current/R mappings must equal the frozen actual input. Positions, actual masks/RoPE and SDPA arguments remain those of the native request.

In all four scientific corners hold P at B_P. At K2 L1 inject:

| corner | Q input | R input |
|---|---|---|
| H00 | E_Q | E_R |
| H10 | B_Q | E_R |
| H01 | E_Q | B_R |
| H11=N | B_Q | B_R |

Run the actual complete K2 computation and head. This is not a splice at L9/L18. H00 is a prefix-contextualized hybrid, **not natural K1**. H11 must equal fresh native K2 exactly.

Technical identity: replace the entire K2 input by E, with original positions/mask/no-cache and full suffix, and compare to natural K1 complete logits. This identity is expected from the pinned no-cache shared-layer source, but must be demonstrated on the technical cases rather than presumed; if runtime loop dependence prevents it, stop and explain before scientific execution. The E-all replay is an anchor, not the H00 corner. Report F(H11)−F(H00) and F(H00)−F(K1) separately; their sum equals the endpoint difference but is order-dependent bookkeeping, not an explained percentage. Keeping B_P means the four-corner claim is conditional on inherited prefix state.

## Four edges and eight controls: twelve boundary endpoints total

For F equal to production gold-minus-mean-nongold gap, retain all four conditional edges:

- Q at R=E: F(H10)−F(H00), forward E_Q→B_Q.
- Q at R=B: F(H11)−F(H01), natural-context rollback B_Q→E_Q with loss sign.
- R at Q=E: F(H01)−F(H00), forward E_R→B_R.
- R at Q=B: F(H11)−F(H10), natural-context rollback B_R→E_R with loss sign.
- Interaction J=H11−H10−H01+H00.

For each of these four directed operations add exactly two controls, all confined to the same region:

1. **Receiver-norm-matched donor:** per token rescale the actual donor vector to the actual receiver vector L2 norm in FP64, cast once to BF16. Save nominal and actual doses and post-cast norms. A nonzero target with zero donor direction is undefined, explicitly recorded, never silently imputed. Zero/zero is identity. This controls norm while retaining donor direction; it does not equalize every geometric property.
2. **Unrelated direction at matched displacement:** independently seed a Gaussian direction by the frozen string `AB1-v1|split|id|edge|absolute_token_position`, normalize it in FP64 to the actual donor-minus-receiver displacement norm for that token, add to the receiver and cast once. Use a specified seed derivation/PRNG frozen before effects. Zero displacement is identity. This is a random-direction control, not an unrelated-question semantic donor. No gold-dependent choice or resampling for an attractive result. Save vectors/cast residuals or their exact reproducible construction and resulting bytes.

P and all nonintervened token rows must remain bitwise receiver-native at the injection. Controls are edge-specific: do not reuse a Q control as an R dose comparator. The two opposite backgrounds are distinct computations; mechanically returning an original input to H11 is an identity, not independent evidence of both necessity and sufficiency. Do not rank Q versus R importance merely by raw effect magnitude: their dimensions, perturbation doses and geometries differ.

## Two Q9 crossed conditions: a real boundary-to-Q9 path test

Let U=H01, the natural K2 receiver with only boundary Q reset to E_Q. Save the actual full L9 input under N and U, including Q and R separately. These are newly generated Q9 states, not the old same-layer K1 donor used by E3/C1.

Run two additional suffix conditions, preserving the boundary intervention where present:

- U with only its actual Q9_Q replaced by N's Q9_Q; P9/R9 remain U's actual receiver values.
- N with only Q9_Q replaced by U's Q9_Q; P9/R9 remain N's actual receiver values.

Together with N and U these form background×Q9-source four corners. Recompute L9 through the complete K2 suffix. Report restoration, clean-background transfer loss and their background interaction. This isolates an actual Q-state mediator from the R bypass accumulated before L9; restoring all of Q9+R9 would be a structural suffix reset and is not a discovery. These two cross interventions remain potentially incompatible mixed states; effects are conditional and cannot by themselves establish a unique natural pathway. No new L10–16 clamping or C1 branch-condition expansion is included.

Total scientific endpoints: four boundary corners + eight edge controls + two Q9 crosses =14. Native K1, E-all and self-patch anchors are separately accounted for. If a component is undefined, retain the question and report the exact defined set; do not replace the question.

## Captures and Experiment A without refitting

Capture the true E/B boundary and selected actual R states at L1 entry, L9 entry, L18 entry and final post-norm. At L10–16 record actual R attention increments; at L18 record actual x/a and selected post-add state. For N/U save actual Q9_Q needed for the crosses. These captures diagnose where inherited-state manipulations first diverge at the selected sites; they do not claim the earliest change among all24 layers or all positions.

Apply the recovered original E2 heads only at their original L18-input and post-norm R sites. Primary descriptive transfer uses the original raw K1-trained head, with its existing unit-L2 and permuted-label controls; all16 frozen applications may be retained as an explicitly secondary fixed table, with no winner selection. No fitting/scaler update or hyperparameter selection. The B_R boundary is K1 post-norm and supports the matching frozen post-norm decoder. Do not apply an L18-trained head to Q9 and call it validated decoding. If showing native-head intermediate lenses, label them diagnostic.

Record offered-order candidate mapping, the E-token zero-positive-label caveat, full frozen probabilities/logits, native/probe decision overlap, per-gold-label/candidate-set strata, scale/dose and natural groups. A causal perturbation that changes fixed-decoder output without changing native decisions is a counterexample to equating accessibility with use. Failure of a frozen linear decoder is not evidence of information absence. BF16 target-logit exact signs and production-score arithmetic must remain separate.

## Cohorts, freeze and inferential rules

Use complete Train non-demo1094, with the existing64 IDs as an explicitly historically selected development/technical partition and remaining1030 as the new-operator confirmation partition; fresh same-site outcome groups need not remain16 each. Do not choose replacement IDs or favorable examples. All four natural groups remain. Validation299 is separate historically exposed confirmation. No new Test experiment or model selection is authorized.

Freeze the source, input/region manifests,14 conditions, controls, analysis and selected decoder keys after a small fixed technical set and before discovery effects. Freeze the interpretation/confirmation claims after complete64 analysis and before reading new1030/Val treatment effects. All14 conditions remain fixed; failure does not authorize scanning layers, heads, token spans, alternate decoders or new control seeds. Root may schedule complete1030 then299 once the freeze is written, without another routine Astra review.

Primary endpoint is question-equal gold-minus-mean-nongold production gap. Also retain every candidate logit/score, fixed natural-K1 wrong gap, dynamic strongest-wrong gap, predictions, ties, rescue/damage/wrong-to-other-wrong, and exact BF16 Fraction signs. Report all natural groups and their natural-no-tie subsets; post-treatment no-tie selection is only secondary sensitivity.

In each confirmation split estimate the four directed edge effects, their norm-control effects and their raw-minus-unrelated contrasts:12 signed mean contrasts, question-paired bootstrap5000 seed20261114 with one-sided Bonferroni family lower bounds at .05/12. Preserve two-sided95% intervals too. A specifically supported inherited-Q route requires both Q-background edge means, norm-preserving directions and specificity lower bounds positive, with no natural-no-tie mean reversal; analogously for R. This is a stringent route label, not a reason to suppress partially supported effects. Require both confirmation splits for a replicated project-internal route. Primary group is fresh natural rescue; other groups remain explicit counterevidence, not exclusions from coverage.

The two Q9 path effects have a separate prespecified two-contrast family (.05/2); require both positive in each split, no natural-no-tie mean reversal, and an upstream Q effect beyond its controls before claiming the tested boundary-Q→Q9→decision connection. An interaction is estimated without a required sign. This path conclusion remains conditional on mixed-state compatibility; the current14-condition order does not establish Q9-specific matched-control necessity or unique mediation. Single-question strict claims require no relevant ties, both signs and documented controls; report the complete counterexample set, not selected successes.

R-only support with weak Q support favors inherited readout-state processing under the tested reset; Q support with weak R support favors contextualized current-input access; both and interaction support distributed dependence. Q boundary loss without Q9 restoration/transfer points to an earlier R bypass or incompatibility rather than a proven Q9 route. These outcomes do not choose between all possible semantic integration mechanisms. Near-zero effects are inconclusive unless a separate equivalence margin was frozen; no posthoc equivalence claim.

## Technical acceptance, budget and stopping

Terra implements CPU nontrivial causal24-layer oracle and an independent injection/replay path. Fix three technical Train IDs by input-only hash order before effects, including a longer input by a deterministic length rule. Verify full-sequence B→K2 identity, E-all→K1, nonintervened rows, actual loops/positions/mask/RoPE/cache, gold-free inputs, complete logits and production scoring. No change of backend to force bitwise agreement inside the primary run. A new hardware site requires fresh N/K1 and source/runtime receipts.

Independent audit separates actual model replay from saved-logit arithmetic; both are required at the frozen technical cases, and full-cohort identity/source/control/score coverage is required. Use exact numerical control formulas, preserve every failure and do not relax thresholds after seeing scientific signs. Reuse C1 source/collector mechanisms where appropriate, not its results as new identities.

Budget is14 scientific endpoint trajectories per question, with native/source capture and technical anchors separately counted: at most19,502 scientific trajectories across1393 Train+Val questions before audit. A two-loop full forward is not a one-loop suffix; report real work. Measure synchronized runtime/peak/storage on the three technical cases and forecast the64/1030/299 stages from it before launch. No invented wall-clock estimate, speculative latency gain, VRAM reservation or interference with unrelated jobs. Full K/V need not be stored; retain the actual intervention source boundary/Q9 states and essential downstream states with lossless compression and byte hashes. Use verified content-addressed deduplication, not presumed prefix equality. If disk space cannot retain the declared evidence, solve storage/verified archival first rather than discarding the intervention sources.

Stop on identity/source/coverage failure, unknown region mapping or unsupported technical semantics. Scientific failure is a reported outcome, not permission to broaden the experiment. Do not repeat C1, E6/H2/Test, train model weights or launch an inference policy search. After this finite experiment, return only a genuine unresolved scientific choice or final result to Astra.

## Reviewed sources and honest scope

Directly read: user attachment; current local C1 NATURAL_MEDIATION_RESULTS/evidence_packet C1_DIRECTOR_INPUT and numerical8 review; RECURRENT_BOUNDARY_SOURCE_AUDIT and pinned modeling_ouro.py boundary/layer code; E2 INFORMATION_AVAILABILITY, recovery audit/status and frozen prediction adapter; E1_E2_RESULTS_AND_SCOPE and final E1–E5 synthesis; E6A baseline report, E6B scientific self-review/state-availability audit and actual natural-feature collector; H2 scientific assessment. Prior completed E3 scientific review remains applicable. This is a focused scientific/source review, not a fresh independent replication of every E1–E6 row, bootstrap, raw tensor or implementation. Terra must reconcile the exact1693cd starting tree and newer C1 source hashes in the execution manifest before launch; no unverified historical artifact is deemed present because its path is named.
