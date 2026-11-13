# E3 Astra scientific review — final decision for this work package

## Verdict and review scope

**The specified L10–L16 attention reference-difference bundle passes its frozen, project-internal Train1030 and Validation299 confirmation. This supports a reproducible conditional upstream-state-to-R-update causal account. It does not identify a unique writer, the initial source of information, or the complete natural K1→K2 mechanism.** The L9-only candidate failed its discovery rule; Stage B and routing tests were correctly not executed. This bounded work package is scientifically closed at that result, with positive and negative findings retained. This review authorizes no additional experiment.

I read the completed L-only summary, SOURCE manifest, all 1329 compact question records, the final decision packet, and the synthesis/path descriptions. I independently recomputed the six production-score contrasts from the candidate scores: maximum disagreement with the supplied rows was 8.88e−16. I also checked group means, paired decision counts and exact-Fraction counterexample counts. The documented bare-model forward, per-site control-mathematics and 5000-resample bootstrap audits were performed by the two Terra executors; I rely on those audits within their stated scope. I did not repeat model execution or the bootstrap, inspect every implementation file, or perform a comprehensive manuscript review. This is not an external blind replication.

## 1. What has been causally established

N is same-site natural K2. P replaces only the Q-region hidden input at physical L9/index8 in K2 with the same-question K1 reference. Q is the entire current input after the fixed demonstration prefix and before the last readout token R: it includes the question, options and formatting, not just a semantic question stem.

At physical L10–L16/index9–15, the intervention adds or subtracts the fixed, same-question N−P reference difference of the actual R attention increment (`input_layernorm_2` output). It applies that difference to the current receiver increment with the specified BF16 cast, then recomputes the downstream computation. It does not clamp the trajectory to N, and its later reference differences already include feedback from earlier altered R states.

Adding this bundle to P improves gold-versus-mean-nongold competition; subtracting it in N reduces that competition. The directions survive the specified norm-preserving controls and exceed the deterministic unrelated-direction effects on average in the prespecified natural-rescue groups. Thus the result goes beyond a latent-distance correlation or an accuracy-only observation: a Q9-conditioned set of later attention updates is causally effective in both tested receiver contexts.

However, subtraction of a counterfactual reference difference is not deletion of a uniquely existing natural component. The warranted claim is a controlled natural-context loss and perturbed-context restoration, not necessity/sufficiency of a complete naturally isolated pathway.

| Frozen contrast | Train rescue337 mean [95% paired bootstrap CI] | Validation rescue104 mean [95% CI] |
|---|---:|---:|
| Restoration | 1.5544 [1.4301, 1.6787] | 1.5345 [1.3025, 1.7674] |
| Natural-context loss | 1.7027 [1.5828, 1.8240] | 1.7471 [1.5315, 1.9559] |
| Norm-preserving restoration | 1.4840 [1.3693, 1.5985] | 1.4611 [1.2436, 1.6747] |
| Norm-preserving loss | 1.7013 [1.5822, 1.8224] | 1.7214 [1.5082, 1.9281] |
| Restoration minus unrelated effect | 1.5732 [1.4261, 1.7241] | 1.6567 [1.3722, 1.9485] |
| Loss minus unrelated effect | 1.2667 [1.1398, 1.3989] | 1.3629 [1.1553, 1.5662] |

All six lower bounds are positive. Natural-no-tie group means (306/100) and exact BF16 target-logit Fraction means retain the directions. These are the registered group-level results; they are not simultaneous statements about every question or every group. The cohorts had previously been used for the parent bundle: this is nested internal confirmation of new interventions, not an untouched subject-level holdout. Hardware-local native groups and baselines must remain separate from A6000 results.

## 2. Findings that constrain the interpretation

**The clean first-read hypothesis was not supported as the selected sufficient account.** L9-only discovery loss was −0.01111 and norm-preserving loss −0.04167. Exact full-shape K/Q/V endpoint bridging verified the implementation, not a positive L9 scientific effect. That failure does not prove L9 is irrelevant: it rejects the chosen bidirectional reference-difference criterion for that factor. L10–L16 remains a multi-site bundle, not seven individually identified writers or proof that every member is necessary. The earlier MLP-reference candidate's failure likewise does not show downstream MLP computation is dispensable.

**The effects are heterogeneous.** Train damage38 restoration/loss means are −0.1946/−0.3043; Validation damage9 means are −0.5787/−0.7292. Stable-correct means are strongly positive, whereas stable-wrong means are small. The supported update therefore does not uniformly move every task toward correctness, and rescue-group membership itself is defined by natural outcomes.

In no-tie rescue questions, at least one of the six contrasts is nonpositive for 63/306 Train questions under production floating-point signs, but **65/306 under exact BF16 Fraction signs**; Validation is 14/100 under either convention. The additional Train IDs are `AKDE&ED_2012_4_21` and `Mercury_SC_415540`: their loss-minus-unrelated contrasts are exactly zero in target-logit arithmetic but +7.45e−8 and +2.98e−8 in production scores. This does not overturn the frozen mean criteria; it prevents false per-question specificity claims.

The same-ID combination of P→P+L incorrect-to-correct and N→N−L correct-to-incorrect occurs in 31/1030 Train and 12/299 Validation questions. Of these, 29 and 10 respectively belong to natural rescue. These counts must not be presented with the wrong denominator, as universal bidirectional mediation, or as an explained percentage of natural loop benefit.

## 3. What the evidence does not identify

The Q splice establishes an upstream dependency, and the R increment interventions establish a causally effective downstream update family. They do not yet isolate the intervening computation that carries that dependency. Later attention differences can combine Q reading, query changes driven by old R state, routing/gain changes and recurrent interactions across layers. The controls reject the tested norm-only and unrelated-direction alternatives on the registered means; they do not exhaust all such mechanisms or distribution-shift explanations.

The R-only L intervention also has a useful structural limitation: with the audited causal/no-cache setup and R at the final position, it cannot change earlier P/Q states. Within a fixed N or P background, it can change R queries, residuals and their suffix processing, but it cannot retroactively generate different earlier Q keys. This constrains a future path explanation without choosing between its remaining R routes.

The parent full-bundle experiment genuinely manipulated the L18 K_Q→R edge. Its posthoc rescue means for key-at-P/key-at-P+A/interaction were Train +0.494189/+0.610318/+0.116129 and Validation +0.437600/+0.569912/+0.132312. This is actual edge-intervention evidence, not merely a distance diagnostic. It is nevertheless a different bundle, a posthoc summary without the new L-only mediation contrast, and not a preregistered CI claim. It cannot close the new L-only path by transitive inference. L18 x/a/SDPA distances also do not demonstrate mediation or restored hidden-state identity.

“R writing” here means experimentally effective changes to R-position updates in a defined interval. “Initial information generation” would require identifying how the relevant earlier Q/R states arose and how that information is used along a specific natural computation. Neither a unique semantic writer nor that initial generation has been established. The three-example FP32 sensitivity set contained no rescue case and cannot establish rescue robustness or general floating-point invariance of this localization.

## 4. Relation to E1, E2 and the separately completed E4/E5

E1 supplies a local SDPA interaction intervention and a finite-precision-aware separation from whole-suffix non-additivity. It motivates a computational account but does not imply that the present L bundle produces that component. Preserve its registered negative cases rather than treating E3 success as their repair.

E2 establishes decodability and separately measured state-intervention effects. Similar probe/native accuracy does not imply identical decisions or native use. The global probe span, offered-candidate span and unequal perturbation doses retain their documented distinctions. E2's residual/attention factorial is compatible with multiple routes, not proof of the L-only route.

Old E4/E5 add separate natural-receiver rollback/restoration evidence for subsets, together with failed specificity/mean/depth generalizations. They use different operators and, in places, different cohorts and hardware. They neither validate this new L bundle at K3/K4 nor demonstrate its downstream mediator. The released E1/E2 (`6c264c35`) and E4/E5 (`ab96019`) evidence should be linked as complementary scoped results, not pooled as one continuous validated causal chain.

## Principal remaining gap and one unexecuted suggestion

The most immediate missing edge is **which downstream R computation carries the L-only effect**, before any claim about initial information origin. A minimal future discriminator would use the actual L18 R residual and attention increment from N and U=N−L and evaluate their two missing crossed combinations, with subsequent MLP/suffix recomputation. Together with existing NN and UU endpoints, this could separate attention-mediated change, residual bypass and compatibility interaction for this exact L subtraction. Restoring the entire L18 x_R would be a structural anchor, not an independent discovery. Such a test would still be controlled mediation, would require an explicitly frozen cohort/controls and would not identify the Q information's origin. **It is a recommendation only; no execution or further expansion is authorized here.**

## Evidence identifiers

- L-only summary SHA256: `ab019e5f7840a4f970a62ba816ef6a2e2fc1f9eed01196bb85fb15a04f3bf757`.
- SOURCE manifest SHA256: `b6fcc97eae9706fc72079366c77eb096172dccbe581db0be714a3e1d0d737175`.
- Complete 1329-row file SHA256: `105044162ab2dcb8cad772a18a38f5dbf5b6db332f772f28fb43e1accbbec67b`.
- Conditional scientific order SHA256: `94469c7fe667e7bf46bf5bb03117436d8163a08987144f0331d34c5a20b9bb00`.
- Execution/audit chains and compact results: `results/conditional_l_holdout/`; parent edge posthoc results: `results/parent_edge_factorial/`; failed E and positive L discovery: `results/factor_discovery64/`.

The earlier PENDING wording in the synthesis/path documents must be replaced by this completed status during publication. This scientific decision is final for the present bounded package; routine packaging and incremental publication remain with Terra without another Astra gate.
