# E4 causal protocol, v1

Final execution status: all fixed E4/E5 designs throughOrder006 are complete and audited. This file preserves the original prospective protocol;later numbered orders extend it without changing past experiments. FINAL_RESEARCH_REPORT.md contains the current conclusion and unlocated-writer boundary.

This is the initial pre-execution protocol, retained for provenance. Its issuance-time status below is historical. Completed results are in the E4 result reports; later frozen follow-ups are orders/ASTRA_ORDER_003_NATURAL_K2_QUERY_FACTORIAL.md and orders/ASTRA_ORDER_004_R_STATE_MEDIATION.md.

Historical issuance status (superseded by completed Orders 001–006): frozen prospective local-causal design; no results existed at issuance. Author:Astra. E1–E3 were in progress in another chat. Read-only audit on 2026-11-12 found completed eight-corner results, not a completed E3 pathway. This protocol tests a candidate downstream mechanism and is not a substitute for upstream mediation.

## Fixed identity and computation

ByteDance/Ouro-1.4B revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1; checkpoint SHA256 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. BF16 SDPA, eval, batch1, no cache, original 25-shot ARC joint-option letter scoring. Hash checkpoint, source, IDs, target token mapping and prompt tokens; retain CUDA/PyTorch/GPU and actual attention arguments. New 4090 environment must not assert bit identity with historic A6000 without measuring it. Keep old and new natural predictions separate, report discrepancy IDs and score deltas. Within-run causal pairing always uses the recipient's current native baseline.

For adjacent loops a,b=a+1, define O_t=SDPA(Q_t,K_t,V_t)_R at L18. Rollback R_ba replaces only the native b-loop SDPA output R row by SDPA(Q_b,K_a,V_a)_R. Restore R_ab replaces only native a-loop R row by SDPA(Q_a,K_b,V_b)_R. All other output rows remain native to recipient. The actual receiver includes its native residual, local norm, MLP and later layers. Compute original full SDPA shapes to preserve numerical semantics. Record injected vector, actual receiver vector, delta, norm and cosine. These interventions modify the readout's access to earlier-loop KV; they do not identify the original token or upstream computation producing KV.

## Competing explanations and controls

H1: adjacent-loop KV updates have decision-specific, receiver-compatible effects. H2: effects are explained by perturbation magnitude/output scale. H3: cross-loop mixed states have nonspecific or receiver-dependent effects. H4: later small accuracy gains conceal substantial but canceling decision changes.

For each direction use the following conditions at identical SDPA_R site:

- Identity: exact recipient O_t reinjection; require zero current-baseline discrepancy (otherwise stop and debug).
- Target: the KV-derived output above.
- Norm-only: O_t rescaled to target-output Frobenius norm; zero norm is flagged rather than divided silently. Controls total amplitude, not direction.
- Unrelated matched delta: O_t + ||target-O_t|| * d_other/||d_other||, with d_other from a different question's same direction KV perturbation. Donor fixed label-blind by cyclic order of frozen IDs. Same R-output shape allows different prompt lengths. Record donor ID and norm before/after BF16 cast. Zero donor delta uses next nonzero cyclic donor and records fallback; if none, unavailable, never fabricated.

Unrelated controls match change norm and norm-only controls match endpoint norm; neither guarantees in-distribution activations. Report both. Gold labels may only calculate metrics and retrospective natural groups, never choose patch direction or donor. Freeze pilot control donor map before results, then freeze Validation map by sorted ID before its run.

## Outcomes and analysis

Store full candidate logits/scores, gold index, first-index argmax and exact top ties for every native and intervention condition. Primary continuous metric is gold minus mean(all nongold); secondary metrics are gold minus fixed wrong (recipient native argmax among nongold, frozen per direction) and gold minus dynamically strongest wrong. Also provide globally K1-fixed wrong sensitivity across depths. Report signed mean, median, positive/zero/negative counts and paired intervention-minus-native deltas. Native four-group stratification is rescue/damage/stable-correct/stable-wrong for each adjacent pair, with K1/K2 stratification separately retained for cross-depth comparisons.

Rollback: among native adjacent-loop rescue questions, report how many revert to wrong, retain correct, improve margins or fail to worsen margins. Also quantify damage among stable-correct, effects on naturally wrong questions, and all-group rescue/damage. Restore: report rescue and damage in earlier recipient, plus paired restoration-versus-rollback patterns by ID. No ratio of transplant rescue count to native rescue count is a mediated fraction.

Every result table contains split, N requested/available/executed/reused/excluded, receiver loop, donor loop, site, condition, correct, rescue, damage, net, ties, all three margins and cost scope. Enriched train pilot does not estimate full-train accuracy. Validation full cohort is the first population-sized confirmation; prior exposure excludes blind-test claims. Preserve all rows and exact counterexample selection rules: rollback improves; restoration harms; positive meanwrong delta but negative dynamicmargin delta; target fails to exceed unrelated; later rescue; later damage.

## Gates and numerical checks

Technical advancement: exact same-process native identity, zero non-R modification, expected loop count, finite tensors, no missing/duplicate IDs, correct checkpoint/source/input identities, control donor differs, required rows complete. Independently recompute scoring/gaps/ties from raw logits. Compare BF16-exact candidate differences with FP32 score arithmetic and count sign/tie changes. This is scoring sensitivity, not a full FP32 forward. On a small frozen subset (first two IDs per pilot group), execute FP32 SDPA_R hybrid recomputation from captured QKV and compare injected values and terminal candidate outcomes; name precision scope explicitly. Full model FP32 only if feasible and separately recorded.

Scientific advancement: do not promote to natural-path mechanism solely because rollback hurts and restore helps. Need actual E3 target, provenance, upstream intervention, measured L18 mediator change, and downstream rescue under the upstream block with target/non-target controls. When E3 arrives, append a new numbered order before running its specific mediation contrast. A direct KV patch followed by resetting the same SDPA row is a deterministic implementation check, not independent causal mediation evidence.

Proceed with negative results and report them. If identity fails, stop that executable branch; do not loosen tolerances post hoc to make it pass. Resource gate uses measured pilot wall time, peak allocated/reserved, CPU RAM and disk forecast. Report omitted coverage with reasons.
