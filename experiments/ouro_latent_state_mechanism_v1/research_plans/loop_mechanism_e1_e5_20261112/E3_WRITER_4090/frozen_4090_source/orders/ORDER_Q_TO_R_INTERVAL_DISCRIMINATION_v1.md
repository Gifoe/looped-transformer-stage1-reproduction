# Decisive order: Q-conditioned R updates in L9–L16

Status: execute by the two Terra agents using mutual code review and numerical audits. Routine implementation, technical acceptance, collection and analysis do not require further Astra approval. Keep the existing E1 Train–Validation–Test chain running; coordinate short owned GPU windows with its shared lock. Do not interrupt other jobs. Do not resume HF bulk upload or direct the other server to run a new writer experiment.

## Decision and its limits

Use decision packet SHA256 `244b920de7c46724b62607e620189e9799b8de7d8f161a854e6454db5c45834e`, its linked complete Q/R discovery64 summaries, and the accepted Terra source/replay audits. Do not repeat those audits or load their full tensors merely to authorize this order.

The candidate is **the Q-conditioned difference in R-position attention or MLP residual increments over loop2 L9–L16**, not a declared semantic writer. At the L9 Q cut, the starting R input is unchanged; the later R state changes substantially, while restoring only the L18 key edge restores part of the effect. This makes a shared Q-to-R route more discriminating than another unmatched R-cut comparison. The small L9-cut loss relative to a late R cut does not establish regeneration of the original hidden state or importance of this interval.

Use only the same frozen 64 Train non-demo IDs for discovery (selection SHA256 `6faad196db958744780ee6477af8839c693798b8b24d2a01a2d450c856c720e1`). Four groups of 16 are stratified exploration, not population frequencies. No new cut, head, token, benchmark, dose grid or learned direction is selected in this order.

## Exact sources, site and update

Keep the pinned model/checkpoint, BF16, SDPA, actual masks/scale/GQA, no cache, eval, batch1, original 25-shot input and original full-vocabulary candidate scoring unchanged.

- N: same-question natural K2.
- P: same-question natural K2 with **Q-region hidden states at L9 input/index8, current_ut=1**, replaced by actual natural K1 hidden states at that same physical input. P/R are unchanged at that cut. This is the previously audited Q producer A1 background.
- Q excludes R and uses the frozen exact token/offset mapping. R is the last token.
- For each index l=8,…,15, capture actual R residual increments `a_l` at `input_layernorm_2` output and `b_l` at `post_attention_layernorm_2` output. Both are after their local norms, before residual addition.
- Define fixed FP64 reference differences `dA_l = a_l(N) - a_l(P)` and `dM_l = b_l(N) - b_l(P)`. These are Q-perturbation-conditioned differences, **not natural K2-minus-K1 branch outputs**. Save their source identities and actual N/P captures.

In a treated forward, the currently computed branch output v is changed to `BF16(FP64(v) + sign*d_l)`, only at R, for the selected branch type and those eight physical layers in loop2. Attention and MLP otherwise recompute from their actual receiver states; the non-selected branch is never clamped. Do not substitute a fixed donor output at every layer: this is a fixed reference-difference intervention, and later treated outputs generally do not equal the original N or P outputs. No cross-layer summation is an explanation of final scores.

Compute the addition in FP64 and cast once. Store actual pre/post BF16 R vectors or equivalent reproducible values, actual effective delta and cast residual. A zero reference difference is identity at that site. Immediate removal-and-restoration of the **actual cast delta** is a technical identity check, not mediation evidence. Do not restore by subtracting only the ideal FP64 delta and claim exact recovery.

## Fixed conditions: 17 total per discovery item

Three anchors: N, P, PK. PK is P with clean N K_Q supplied only to the L18 R-query SDPA output row, exactly as the frozen R-edge addendum. Other query output rows remain those of P.

For each branch X in {attention A, MLP M}, run these seven conditions:

1. P plus dX at the eight sites.
2. P plus dX, with each proposed updated output rescaled to the norm of that site's actual pre-intervention v before the final BF16 cast.
3. P plus uX, the equal-reference-dose unrelated direction control below.
4. N minus dX at the eight sites.
5. N minus dX with the same per-site norm-preserving operation.
6. N minus uX.
7. P plus dX plus the clean N L18 K_Q→R edge restoration (PK's mediator definition, with this condition's actual receiver query/V/residual).

For an unrelated source, use a deterministic other question from the frozen64 reference pool, excluding identical ID/input. Order candidates by SHA256 of `q-to-r-interval-v1|recipient_id|donor_id`; never use gold, group, predictions or effect size. For each l and X use that donor's N−P difference, rescaled in FP64 to the recipient's `||dX_l||`. Select the first eligible nonzero donor for that site, log its identity, and freeze the mapping before reading new intervention effects. This allows the same frozen Train pool to serve the later holdouts. If the recipient difference is zero, u=0. If no eligible direction exists for a nonzero reference dose, mark the control undefined rather than fabricate a vector.

For norm preservation, if the proposed vector has zero norm but v does not, mark undefined; both zero gives zero. Record actual BF16 norms and effective per-site doses. Main and unrelated controls match their reference additive dose; BF16 rounding differences remain explicit. Norm preservation is a direction sensitivity control, not proof of semantic content.

No combined attention+MLP intervention is added automatically. A null result for both individual bundles does not exclude a jointly necessary bundle; it rejects this round's individual-bundle claim.

## Technical acceptance and finite execution

Terra_executor owns reference N/P capture, Q-boundary/source bindings and the PK edge; Terra_e4e5 owns branch-difference intervention, independent mathematical/replay audit and analysis. They may redistribute routine work but use one GPU owner/lock at a time.

1. CPU tests independently verify the signed additions, matched reference doses, norm/zero cases, R/loop/layer identity, finite values, cast residual and hook restoration. Include a full-suffix toy; do not merely test the implementation against itself.
2. Reuse the three already fixed technical questions, without interpreting their effects. Verify unchanged N and P full logits against the earlier Q-producer cut8 artifacts; PK must bridge earlier cut8 A1K0. Verify actual starting R equality in N/P and source provenance. Verify non-R states and L18 K_Q remain those of the selected background for all R-branch treatments. For PK composites, document the actual R query and leave other SDPA query rows unchanged. Independent replay reconstructs the saved actual branch injections and full-logit scores. Report whether a check is independent model replay or only scoring.
3. Freeze final source/protocol and the reference/donor ledger; execute all64 and audit before reading any new treatment effects. N/P reference acquisition may precede donor freezing, but no new add/remove/control effects may be used to select donors.

Budget ceiling for discovery: 64×17=1,088 condition forwards, including the two reference forwards reused as N/P, plus fixed three-case technical work and one independent replay pass. No full QKV archive is required; save the eight-layer R references/injections, small target states, logits, hashes and receipts. Prior throughput suggests approximately 4–6 minutes forward-only and roughly 10–25 minutes including replay/I/O under favorable sharing; this is an estimate, not an SLA. Replace it with the three-case measured budget, preserving load/save/wait scopes. Unexpected anchor/source/numerical failure stops that run; retain artifacts and fix a technical cause without dropping difficult cases. If the estimate exceeds 30 minutes for discovery, report the measured reason and reduce redundant anchor work, not the condition or ID set.

## Outcomes and the competing explanations

Let F be the production gold-minus-mean-nongold candidate gap, equal within question then equal across questions. For each X report:

- restoration R_X = F(P+dX)−F(P);
- natural-context loss L_X = F(N)−F(N−dX);
- their norm-preserving counterparts;
- paired specificity R_X−R_unrelated and L_X−L_unrelated;
- late-key increment on the treated background F(P+dX+K)−F(P+dX), compared with F(PK)−F(P), and the resulting difference-in-differences.

Also report fixed K1-wrong and dynamic margins, all candidate logits/scores, first-argmax/ties, exact BF16 target-logit Fraction signs, per-question restoration/damage and the same-ID intersection of bidirectional effects. Report all four natural groups, natural no-tie sensitivity, actual L18 x_R/a_R distances and directions, and fixed-readout diagnostics as diagnostics. Do not derive hidden-state reconstruction from recovered accuracy or use a percentage mediated/explained.

- Attention-bundle restoration plus reciprocal removal beyond controls supports a candidate Q-conditioned attention-update route into R.
- The corresponding MLP result supports a candidate downstream transformation/write route. MLP has not directly read Q tokens; its input can already contain Q-derived information.
- Restoration without natural-context loss supports compensation/sufficiency, not natural necessity. Removal without restoration supports disruption, not a sufficient account of the producer deficit.
- No difference from unrelated directions leaves a generic perturbation explanation. Loss abolished by norm preservation leaves a scale explanation.
- A remaining late-key effect indicates an additional conditional route; a changed key effect indicates compatibility or interaction, not a uniquely additive mediation split.
- Opposite effects in the natural-damage group and no flips despite large margins in stable-correct items are retained as substantive counterexamples.

All findings at this point remain local counterfactual evidence. Do not name an answer-writing semantic module, complete NaturalPath or Mechanistic evidence on the basis of these results alone.

## Prespecified continuation and holdout criteria

Before executing discovery treatments, register the following deterministic rule. For X to qualify for holdout, the discovery natural-rescue group must have strictly positive means for R_X, L_X, both norm-preserving effects, and both unrelated-adjusted effects. Report the natural no-tie versions; a sign reversal there prevents qualification. Undefined controls must be listed and prevent a blanket full-scope claim. Do not rank a winner by accuracy or select a subset of supporting questions.

If neither X qualifies, stop this interval's expansion and return the complete negative/mixed evidence to root as a decisive hypothesis failure. Do not automatically scan layers, add combined bundles, change dose, relabel a writer, or launch a recurrent alternative. The overall research is not declared complete merely because this candidate fails.

If one X qualifies, keep N/P/PK and that branch's seven conditions (10 total). If both qualify, retain all17; do not pick the more favorable mean. Freeze that choice and the exact six directional claims before any new holdout treatment effects. Execute all remaining Train non-demo1030 (excluding the fixed64 and25 demonstrations), then Validation299 with the same references/rule/weights/conditions. These are project-internal prospective checks; previous natural/probe/ARC exposures mean they are not independent blind datasets. No new Test writer run is authorized by this order.

Test the six directions separately in each holdout's natural-rescue stratum; report all other groups without selecting them. Use question-paired bootstrap with 5,000 resamples, fixed seed20261112, and the lower bound of a two-sided95% percentile interval for each claim. A branch is prospectively supported only when all six lower bounds are above zero in **both** Train1030 and Validation, and the natural no-tie means do not reverse. This intersection rule tests a conjunction; the two prespecified branch candidates are retained rather than selecting one post hoc. Report the limitations of the conditional subgroup bootstrap and within-dataset dependence; do not present it as external replication. Report failures verbatim, not only pooled means.

Positive holdout evidence permits a precisely scoped candidate interface (sites, difference vectors, sources, controls, natural-context loss, restoration, state diagnostics and counterexamples) for the next decisive path assessment. It does not authorize additional fine-layer/head/token searches or automatically redirect the other server. Terra performs all routine steps without an Astra gate. Astra returns only for a genuine hypothesis change or final scientific interpretation.

## FP32 scope

The completed three-Train full-FP32 sensitivity examples contain damage/stable-correct/stable-wrong and no rescue. Record their exact scope, parameter promotion and TF32/backend settings. They do not establish rescue robustness or the robustness of this new interval intervention. This order adds no rescue FP32 claim and does not turn the earlier CPU score recomputation into independent GPU-forward validation.
