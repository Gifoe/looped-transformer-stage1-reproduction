# Astra paired-message V2 protocol — 2026-11-17

Prospective new estimand. No new-task neural outcomes have been observed when this protocol is written. V1 remains FAIL_PERMUTATION_NULL; this file neither supersedes its evidence nor makes its native-recompute operator pass. Root may proceed under the existing authorized goal after technical gates. Astra has not implemented or run this operator.

## Scope and correction
The root's intended generator has FOUR options: the two mapping terminals plus two distinct nonpredicted distractor terminals, with label tokens A/B/C/D=[330,389,340,422]. The earlier two-option warning concerned a deficient written specification, not an observed four-option implementation. Freeze the four-option generator explicitly.

V2 adds the ideal real-attention permutation difference to an actual native BF16 SDPA base. It does not claim bitwise equivalence to a full native BF16 counterfactual permutation. Self/joint zero and K/V equality are construction identities, not independent scientific controls. The new bounded claim is graph-directed pair-message sensitivity in a natural round background. It is not unique semantic representation, G-local composition, new information acquisition or the complete K1→K2 algorithm.

## Frozen task and site
S contains two object→box relations. G contains two complete, equal-token-length box→terminal mapping records, including delimiters. H asks about an object and supplies an independent terminal-to-label coding. Querying a box directly is forbidden. The third and fourth offered terminals are distinct and absent as mapping endpoints. All four offered labels are scored; save full49152 logits and ties.

Cross first-hop assignment, endpoint assignment, queried object, label permutation and record order: five binary factors,32 inputs/base group. The two label permutations must be frozen derangements covering the four labels in balanced semantic roles across groups; firsthop/endpoint/record-order flips must not accidentally alter label-role assignment. Graph-derived selected terminal and opposite-record terminal are calculated before inference. Distractors are the other two offered terminal entities. No correctness-based sampling.

For each fixed S/G prefix, selector/label suffix variants must have byte-identical native G states and actual KV in both rounds. Interventions preserve the complete prefix through G. This excludes a single future-selected label in G but not all-answer precomputation.

Only configuration: both whole G records, all heads, all H receiver rows, physical L9–16 inclusive. Actual post-RoPE K/V, actual current Q, full causal prefix. Semantic and random arms separately intervene in K1 or K2. No receiver, head, token or window fallback after outcomes. The broad receiver choice avoids unsupported single-token localization.

## Paired-message operator
At each current receiver/head, use explicit FP64 softmax over every causally visible source with the actual scale and actual current Q/K. For corresponding record positions Aj,Bj,
Delta=sum_j (p_Aj-p_Bj)*(V_Bj-V_Aj).
Production base is the ORIGINAL full-shaped native SDPA result O from this very invocation, not its FP64 reconstruction. Write once: BF16(FP64(O_H)+Delta). Other output rows are copied unchanged. Continue actual residual, projection, MLP and subsequent layers; recompute each later Delta from that condition's current trajectory. Do not substitute native-reference Q/K/V into later interventions. Preserve all source tensor shape/dtype/stride and model settings.

For each semantic arm save per-layer/per-H-row actual BF16 write dose: norm(after-before), flattening all heads and head dimensions within a receiver row. This is a pre-o_proj norm, not a projected-update or residual norm. Save requested Delta and projected actual change as well.

## Random comparator
For every semantic unit/round, random uses the corresponding semantic actual BF16 row doses as targets at matching layer/row. It operates on its OWN current trajectory. Directions are seeded Gaussian vectors, independently keyed by frozen unit/round/layer/row; no seed search. Scale each vector using a deterministic frozen BF16-aware dose matcher. Zero targets write unchanged output. Do not reuse the semantic direction, hidden states or current probabilities in the random trajectory.

Keep every unmatched row and every input. Freeze QC: sum absolute dose error / sum target <=1%, and at least99% of total target-norm mass belongs to rows whose absolute dose error <=max(1e-9,1% of target). When all targets are zero, demand all actual doses zero and report mass criterion as degenerate, not positive evidence. Report row-count fractions, mass fractions, max errors and distributions separately by round, layer and split. Both global QC and each round QC must pass. Failure means semantic specificity against this comparator is unestablished; it does not permit deleting rows or retuning seeds on confirmation. Norm-matched Gaussian directions are not a universal certificate of semantic irrelevance or manifold equivalence.

## Scientific conditions and budget
Five conditions/input: native, semanticK1, semanticK2, randomK1, randomK2. Each returns both round heads. Native is shared. Each intervention run edits exactly8 SDPA invocations, writes exactly8×|H| rows, and still calls original SDPA exactly48 times (24layers×2rounds); explicit FP64 calculations are counted separately. Four intervention arms give32 edited invocations/input. Count actual original-SDPA calls in the wrapper, not layer visits.

16 development groups plus64 independent confirmation groups.32inputs×5conditions×80groups=12800 scientific forwards (2560dev,10240confirm). Technical runs, oracle repeats and runtime benchmarking are separate ledger entries. No speculative ETA; benchmark authorized old technical cases first. Freeze source, generator, seeds, metadata and analysis before confirmation.

## Primary estimands
At each round's OWN terminal head, Y=rawlogit(opposite)-mean(rawlogit(two distractors)); G=rawlogit(opposite)-rawlogit(selected). Decode roles using that input's actual label coding. Same-round K1 contrasts use K1 heads under K1 interventions; same-round K2 contrasts use K2 heads under K2 interventions. K1-arm final K2 heads are secondary because they include an extra computation round.

Seven primary contrasts:
1–4: Y_semanticK2-Y_randomK2, separately in the four firsthop×endpoint strata, averaging selector/label/order equally.
5: pooled Y_semanticK2-Y_nativeK2.
6: pooled G_semanticK2-G_nativeK2.
7: pooled [(Y_semantic-Y_random)_K2-(Y_semantic-Y_random)_K1].

10000 deterministic base-group bootstrap resamples, preserving all32inputs and paired conditions. Bonferroni7 two-sided intervals at99.285714% each. Contrasts1–6 require adjusted lower bound >0.1 rawlogit. All eight firsthop×endpoint×selector point estimates of Y_semanticK2-Y_randomK2 must also be positive. Contrast7 is separate: positive adjusted lower bound supports stronger K2 participation; its failure does not erase evidence for pair-message sensitivity in K2. Publish the entire family, including failures, not a selected successful subset.

Contrast5 rejects a pure selected-logit-only reduction with unchanged opposite and distractors. Contrast6 rejects an equal shift to both mapping terminals with unchanged gap. Contrasts1–4 assess specificity beyond the chosen matched corruption control. These JOINTLY exclude those simple explanations; they do NOT exclude arbitrary structured corruption that differentially changes salient terminals. Report rawlogit changes for selected/opposite/each distractor individually, probabilities and predictions, so the limits are visible. The justified claim remains this finite intervention's graph-directed sensitivity, not a universal proof that only a semantic computation can produce the pattern. No finite random comparator establishes that stronger claim.

## Native capability and stop rules
Development and confirmation each require native K2 accuracy>=70% and a paired base-group-bootstrap95% CI lower bound>0 for native K2-minus-K1 accuracy. This capability gate concerns interpretation of K2 improvement; no filtering of native-wrong or intervened-wrong cases. Failures stop this version's mechanism claim and cannot trigger easier templates or alternative sites. All errors/ties remain. No confirmation-dependent effect threshold, numerical bound, dose matcher or seed changes.

## Technical independent validation before new neural outcomes
Use32 pre-existing worlds, balanced16K1/16K2. Freeze seven conditions/world: native, instrumentation-noop, self, semantic-K formula, semantic-V equivalent route, joint-zero, and random-dose comparator;224 forwards. The additional32 random technical forwards assess the same frozen actual-dose QC defined above; retain all unmatched rows. They do not change scientific thresholds or scientific forward counts. In the K/V verification routes the independent oracle must actually construct complete permuted K/V and recompute full attention, not reuse the paired formula. Scientific production need not duplicate the constructed K/V conditions.

Mandatory exact invariants: noop/self full-vocabulary heads bitwise native; joint-zero full-vocabulary heads bitwise native; untouched non-H output rows bitwise identical within invocation; complete prefix state/KV identity;48 original SDPA calls;exact8edit events; correct arm and receiver/source spans. Explicit zero path must return original bytes, avoiding signed-zero or casting artifacts. Preserve frozen V1 failed results separately.

Capture all8 events on at least4 frozen old worlds (both arms and both templates if available), including full actual Q/K/V needed for each H causal reconstruction, native SDPA_H, requested Delta, actual BF16 after, and pre/post-o_proj differences. Capture per-row scalar audits for ALL32 worlds and all events. Independent CPU oracle uses FP64 full softmax with unpermuted, K-permuted, V-permuted and jointly permuted KV. Validate paired Delta against counter-minus-base and K/V equivalence, and validate joint counter-minus-base near zero. Oracle must include full causal masks, scale and any GQA expansion explicitly. It must not replace production native attention.

Numerical tolerance for independent FP64 oracle comparisons is frozen per receiver/head/vector from dimensions and scale: let n be number of visible source tokens, d head dimension, s=max(1,max absolute V element), eps64=2^-52; absolute elementwise bound tau=128*eps64*(n+d+1)*s. Check maxabs discrepancy <=tau and report discrepancy/tau, full dimensions, values and norms. This conservative first-order computational allowance is an engineering threshold, not a universal formal floating-point proof. Reject nonfinite intermediates; extremely saturated/ill-conditioned cases require diagnosis, not widening tau after outcomes. Softmax row sum error bound128*eps64*(n+1). Request oracle results for native, K,V,joint separately; do not call typical1e-15 an invariant.

Report native BF16 SDPA versus FP64 native reconstruction max/RMS/relative L2 errors per row/head; prefreeze reconstruction relativeL2 <=1% for nonzero rows, with denominator max(norm(FP64base),1e-12). Also report absolute errors near zero rather than hiding them behind ratios. This1% is a conservative inherited engineering check, not proof of equivalence. Fail it and pause for diagnosis. For each event report reconstruction error relative to requested/actual intervention norms without forcing every tiny Delta to exceed native error. Tiny perturbations must not be advertised as accurately emulating a native counterfactual. Verify actual after equals BF16(native+pairedDelta) bitwise and audit the complete projected actual write, accounting for biases consistently.

The oracle tolerance is unrelated to the0.1rawlogit scientific threshold. Constructed zero nulls cannot estimate a noise envelope or serve as scientific controls. Technical failure stops this version; repairs require a new frozen source/version and rerun old technical gates. Preserve failed versions and do not retrospectively widen bounds.

## Remaining identification limit
Even all seven contrasts passing does not distinguish G precomputing every answer from G relaying a table subsequently composed in H. Nor does a stronger K2 effect demonstrate that K1 lacked relevant information. This is a concrete test of semantic pair-message participation under a newly defined deterministic attention intervention. Full mechanism identification remains active and incomplete.
