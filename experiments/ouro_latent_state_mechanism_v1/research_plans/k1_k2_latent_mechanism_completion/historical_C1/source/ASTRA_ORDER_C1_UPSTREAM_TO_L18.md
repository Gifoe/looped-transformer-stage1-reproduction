# C1 frozen order: a validated upstream conditional update to independent L18 branches

Scientific director: Astra. Implementation, all numerical analysis, GPU execution and independent arithmetic/replay audits: current Sol, coordinated by root. Date: 2026-11-14. This is a mechanism experiment, not an accuracy method. No new agent, no E6C, no new Test evaluation, no weight update.

## Decision and scope

The missing edge selected here is **the already supported Q9-conditioned L10–L16 attention-reference update -> actual L18 residual-bypass and attention-increment routes -> K2 decision**. The old L-only experiment is complete on Train1030 and Validation299. Its six directional claims passed, and root has independently checked arithmetic/coverage of all1329 derived rows. Do not rerun it as an allegedly new localization study. Necessary current-hardware source anchors are included below because old4090 tensors cannot serve as exact5090 counterfactuals.

Earlier E4/E2 four corners replaced L18 input with natural K1 x and measured the attention it produced in K2. C1 instead starts with the actual state produced by a specific, previously validated upstream intervention. Neither old experiment closes this edge. Conversely C1 does NOT identify the generation of the update from the true recurrent boundary. N−P below is not natural K2−K1. A positive result still leaves the full-sequence K1-final-RMSNorm -> K2-L1 -> Q9/update-generation edge unresolved.

Do not interrupt or change running E7. CPU implementation and static audits may start now. C1 GPU work enters the same owned queue only after E7 releases it. E7 remains a natural-input/protocol-mechanism control; its accuracy-pair and gate-profile follow-ons are superseded by the user's new mechanism-first command.

## Fixed setting and source definitions

Use ByteDance/Ouro-1.4B revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1, checkpoint SHA25658872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af, current source/runtime with complete hashes, BF16 SDPA, eval/no-cache, original Joint25-shot token IDs and candidate scoring. Bind physical layer and current_ut explicitly. Keep all preexisting source/data files immutable.

N = natural K2. P = natural K2 with only the original Q region at L9 input/index8/current_ut1 replaced by same-question native K1 Q states at that site. Use the audited original region boundaries, not a newly interpreted question stem: Q includes all post-demo, pre-R query material. Prefix and last-token R are unchanged at the cut.

For each l in physical L10–L16/index9…15, capture a_l as actual input_layernorm_2 output at R immediately before attention residual-add. Define d_l=FP64(a_l(N))−FP64(a_l(P)). U = N with BF16(FP64(current a_l)−d_l) injected at each such site, one cast. This is the original conditional reference-difference operator, not a clamp to a_l(P). All subsequent computation is real. Capture fresh same-hardware N/P/d/U; never substitute natural K2−K1 or earlier-device d.

At physical L18/index17/current_ut1, capture actual input x, post-localnorm increment a, native BF16 y=x+a, actual post-MLP state and final normalized readout for N and U. Let their branch references be xN,aN and xU,aU. No final-logit or final-state patch is permitted. Keep the full L18 MLP and L19–L24/recurrent final norm/head computation.

## Crucial implementation contract

For every U-background corner, first execute actual U attention/QKV normally. Then select the residual-bypass branch x and attention branch a independently and perform the same BF16 addition as the source. Thus x restoration must not alter the attention query/key/value that generated aU. A pre-attention x replacement is a different intervention.

The model saves y as the MLP skip before calling post_attention_layernorm. Returning a new tensor from that norm's prehook changes only the MLP input and silently leaves the skip wrong. Use a source-locked explicit wrapper, or an audited same-storage in-place R-row update that changes both aliases. Independently prove both MLP input and MLP skip equal the intended y, and that non-R rows and prior attention are unaltered. Do not implement residual selection as a+(xN−xU): BF16 reassociation is not equivalent.

Native branch identity must reproduce the complete full-vocabulary endpoint hash. Save before/after branch values and actual y/MLP skip; preserve reference tensors by cloning before any in-place write. Audit execution counts, current_ut, sites and repeated hooks; no stale captured tensors.

## Cohort and execution

Freeze all original1094 non-demo Train and299 Validation IDs/token hashes before any C1 outcome. Reuse the original64 non-demo discovery IDs only as an engineering batch, not an equal-population estimate or a new independent discovery cohort. The other1030 Train and299 Validation remain fixed. No result-based exclusion, sample replacement or conditional expansion. All four fresh native K1/K2 groups are retained and reported separately. Previously exposed ARC splits are project-internal replication, not unseen data.

Two fixed technical items are the lexicographically first two original discovery IDs. After technical acceptance, run the64, then1030, then299 with exactly the same frozen operators regardless of scientific sign. A code/source/identity failure stops the affected namespace for repair and new versioning; a negative scientific result does not trigger a changed hypothesis or a favorable subset. Report measured cost before the full queue; the user's lack of time limit does not authorize unrelated scanning.

For mediator unrelated donors, sort IDs lexicographically within Train1094 and within Validation299 separately and take the next ID cyclically. Freeze maps and hashes before outcomes; donor selection never reads gold/groups. Donor mediator delta is its own fresh N−U at the same branch, with the same model/source. For upstream unrelated anchors preserve the old L-only rule if fully reconstructible; otherwise freeze a clearly named new sorted-ID control and never call it an exact old-control replication.

## Main endpoints and controls (17 endpoints/question plus native K1)

1. N, 2. P, 3. P+L using the original fixed d_l addition, 4. U=N−L.
5. UX: U upstream retained; L18 branches xN,aU.
6. UA: U upstream retained; L18 branches xU,aN.
7. UXA: U upstream retained; L18 branches xN,aN.
8–9. U with the original per-site norm-preserving subtraction and equal-reference-dose unrelated subtraction, respectively. These anchor upstream specificity on this hardware; norm preservation is NOT equal dose.
10–13. In U, restore each branch independently with (a) norm-only and (b) unrelated-delta controls: X_norm, X_unrelated, A_norm, A_unrelated. The other branch remains its actual U value.
14–17. In N, roll each branch back independently with its corresponding norm-only and unrelated-delta controls; the other branch stays native N.

Norm-only for restoration of vector vU to vN means retain vU direction and scale to ||vN||, using FP64 norm/scale and one BF16 cast. Rollback reverses the roles. Unrelated restoration adds the frozen donor's N−U direction scaled to ||vN−vU|| to vU; rollback subtracts that same scaled direction from vN. Scaling/addition use FP64 and one BF16 cast. Record requested versus actual after-cast dose, endpoint norms and cosine. Explicitly mark undefined zero-direction cases, retain their other conditions, and report coverage; never invent a direction. Attention and residual branches use separate donor deltas/norms. If reference target delta is exactly zero, the identity control is valid and is labeled zero dose.

On technical items and the fixed independent replay subset, also execute reverse target corners in N: xU,aN must equal UA; xN,aU must equal UX; xU,aU must equal U. These are bridge identities, not extra independent samples. Because R is last and U only changes R, all non-R L18 and later states must match N. UXA must reproduce N full logits exactly. This is a structural endpoint anchor, not evidence of a discovered percentage of mediation. If any equality fails, resolve the implementation/source premise before interpreting effects; do not weaken equality after seeing results.

## Estimands frozen before outcomes

Let F00=F(U), F10=F(UX), F01=F(UA), F11=F(N)=F(UXA).

- Total upstream loss: F11−F00.
- X restoration in U: F10−F00; reciprocal X loss in N: F11−F01.
- A restoration in U: F01−F00; reciprocal A loss in N: F11−F10.
- Interaction: F11−F10−F01+F00. Report both conditional effects of each branch. Do not divide by total loss or call these natural indirect-effect percentages.
- Each branch's target restoration minus its norm-only restoration, and minus unrelated restoration; analogously target rollback loss minus each control loss. These contrasts, not raw target magnitude alone, test specificity.

Primary F is full-vocabulary FP32-logsoftmax gold-minus-mean-nongold candidate gap, question-equal. Preserve raw BF16 candidate logits, logsumexp, all scores, gold-minus-fixed-K1-best-wrong, dynamic gold-minus-best-wrong, accuracy, first-argmax, complete top ties and exact BF16-logit Fraction signs. Best-wrong excludes gold even when K1 predicts gold; it remains fixed per question across interventions. Gold is only used in offline analysis, not in intervention/donor selection.

Natural rescue is fresh same-hardware native K1 wrong/K2 correct. Report the primary rescue-group means and all-cohort/four-group tables separately, including no-natural-tie and all-endpoint-no-tie sensitivities. Report discrete losses and restorations on the SAME IDs: N-correct/U-wrong followed by each restored corner, controls, and both-direction intersections. Strict branch cases require no relevant endpoint ties and neither norm nor unrelated control making the same successful reversal; list failures as well as successes. Do not equate gap loss with decision necessity or restored counts with a mediated proportion.

Sol computes question-paired bootstrap10000, seed20261114, separately for64,1030, full1094 and299. Fixed groups are resampled as questions within that group; this is not model-seed replication. All primary means/CI and control-adjusted contrasts are reported, with no threshold tuning. Strong branch-route support requires positive restoration AND reciprocal loss beyond BOTH controls in1030 and299, with simultaneous one-sided95% lower bounds across the12 route tests (2 branches × 2 directions × raw/norm-adjusted/unrelated-adjusted), e.g. Bonferroni percentile bounds alpha=.05/12, and no sign reversal in natural-no-tie means. Ordinary paired95% intervals remain in tables. Failing this gate leaves descriptive/case-specific evidence; it does not erase the actual results or authorize site search.

## Independent and numerical audits

Independent replay subset: the first8 Train IDs by SHA256('C1-replay-v1|'+ID), fixed before outcomes, plus the2 technical IDs (deduplicate). Use an independent injection implementation to verify actual y/skip and complete endpoints for the core four corners and UXA; source/replay provenance must be distinct and explicit. Reusing the producer helper is not an independent implementation.

All rows receive independent score/tie/exact-Fraction/control arithmetic audit and source/input/capture hash binding. Save compact actual L10–16 before/after/reference/cast residual, L18 x/a/y/post-MLP, QKV_R and actual SDPA_R, final-normalized R, candidate/full-logit hashes, full original attention argument identities and non-R hashes. Full QKV need not be archived. Score equality is not enough for hook correctness.

Numerical sensitivity is fixed to8 Train IDs: first4 hash-sorted fresh natural-rescue and first4 hash-sorted other Train IDs using SHA256('C1-fp32-v1|'+ID), based only on native endpoints before C1 outcomes. Recompute the core N/P/U/UX/UA/UXA references and paths in full-model FP32 with its actual SDPA backend recorded; do not transplant BF16 d into FP32. This estimates combined numerical/backend sensitivity, not an isolated dtype effect or an independent population replication. If fewer than4 in a group, include all and record the shortfall. Numerical sign/tie-sensitive examples remain counterexamples, never exclusions from the primary BF16 result.

## What each possible outcome permits next

Both independently specific branches: a distributed conditional pathway with interaction; no unique writer or semantic-answer claim. Only A supported: the L-induced change has an attention-mediated component under these fixed contexts. Only X supported: support bypass into the real MLP/suffix; do not infer that attention is globally irrelevant. Target restoration without reciprocal loss: compensation/sufficiency only. Loss without restoration: disruption or context dependence. Controls equally effective: direction-specific explanation fails. Joint-only recovery is expected structurally and cannot rescue a failed mechanism claim. Strong positive means with no robust discrete cases do not explain decision correction by themselves.

After C1, return one scientific decision packet with all counterexamples. No automatic layer/head/token sweep, probe refit, new inference controller or accuracy optimization. The next unresolved edge is actual recurrent-boundary information generation/use; its next experiment must be chosen from C1 evidence and the existing E2 fixed probes/candidate-subspace controls. Failure of C1 is a legitimate restriction, not completion of the overall research goal.

## Deliverables and ownership

Sol writes frozen input/operator/source/control manifests before outcomes; commands in orders/; complete coverage/cost/failure receipts, raw endpoints and paired question-level contrasts; NATURAL_MEDIATION_RESULTS.md, NUMERICAL_AND_SOURCE_AUDIT.md, COUNTEREXAMPLES.md and machine-readable evidence tables. Root maintains the source index and queue. Astra owns hypothesis interpretation and subsequent scientific order, not routine numerical gates. All final reports must distinguish already completed E3, prior E4, new C1, source audits, and remaining boundary-generation uncertainty.
