# Final E4/E5 research report

Final status, 2026-11-13: all fixed E4/E5 experiments through Orders 005/006 are complete and independently audited. No further GPU intervention is authorized by this review. The remaining generating-component uncertainty is an explicit scientific limit, not unfinished E4/E5 execution.

**Scientific conclusion:** the evidence supports a local R-state/attention readout mechanism with question- and receiver-dependent contributions. It does not support reducing the natural second-loop benefit to the previously observed KV interaction, nor does it locate the computation that originally generated the useful state. The fixed confirmation 128 strengthens the residual/suffix and subset attention-restoration evidence while failing to reproduce strict KV decision necessity.

## Scope and design

Model ByteDance/Ouro-1.4B, revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1, checkpoint SHA256 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. Fixed25-shot ARC-Challenge Joint-Option Letter Scoring,24 shared physical layers, BF16 SDPA/eval/batch1/no cache. Current GPU RTX4090, torch2.5.1+cu124, transformers4.55.0. All interventions use physicalL18/index 17 and actual repeated-loop gating, full-shaped attention and R-only replacement. Labels enter analysis, not current-question intervention selection. Original weights/history are preserved.

The original KV299 followed Train 64 discovery. Later Q/R operators and the late-depth extension were selected after viewing299, so those results are exposed-cohort analyses. Order 006 froze a label-blind remaining-Train 128 by ID hash before its new interventions, excluding25 demonstrations, both discovery64 sets and token-identical inputs in the prior64/299/peer64. There were 975 eligible IDs and 144 excluded Train IDs. The128 questions were not used to choose this task's new causal operators, but had historical full-cohort/eight-corner exposure and may have entered peer probes. This is not a globally unseen or blind test. Cohorts are never pooled.

The sequence was fixed KV cross-loop pilot 64/full 299; natural-K2 Query factorial; actual-R-state/attention four corners; matching K3/K4 state experiments; then preregistered cross-question128 confirmation. New directions followed genuine negative/ambiguous findings, with separate frozen orders and namespaces. The final128 conditions were fixed together and did not trigger further scans. Identity, norm and unrelated controls, all metric definitions, original donor maps, exact source/tensor hashes and numerical diagnostics are retained.

## KV and Query results

On299, forward KV restoration into K1 gives 57 rescues/3 damages; natural K2 KV rollback gives 4/14. Among 104 natural corrections,14 are lost,9 pass both controls and no-native/target-tie requirements. Only one question is both forward-restored and reverse-damaged, and its rollback endpoint is tied: zero meet both directions without ties. The enriched pilot's sole reverse loss is also a tie/control-sensitive case.

On confirmation 128, KV rollback gives 1 rescue/3 damages, versus unrelated2/5 and norm0/0; meanwrong effect−0.296346 versus unrelated+0.028035. Its strict natural-correction necessity count is 0. Query-only gives 2/4 rescue/damage and meanwrong−0.095825; jointQKV gives 0/5 and−0.410360. Each retains3 strict natural-correction losses. Query-direction/headnorm-matched effect−0.092611 differs from the magnitude-only effect+0.001790. These support a subset Query-direction role without showing that Query or KV alone explains natural gains. New Q×joint-KV finite differences use different factors and receiver from historical fixed-K1 K×V; neither directly refutes nor reproduces that older contrast.

## Actual state-to-attention mediation

The four corners compare N, actual state block U, native-attention restoration under U, and A(U) alone. A(U) is not natural donor attention. Raw normalized-increment and residual-sum checks verify the intended causal intervention.

| Cohort | Receiver | Native correct | U correct | Restored-U correct | A-only correct | Natural rescues | Lost under U | Recovered by A | Strict four-control recoveries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 299 | K2 | 243 | 220 | 240 | 235 | 104 | 29 | 22 | 11 |
| 299 | K3 | 258 | 251 | 256 | 253 | 20 | 9 | 4 | 2 |
| 299 | K4 | 261 | 256 | 256 | 263 | 7 | 5 | 1 | 1 |
| 128 | K2 | 103 | 98 | 105 | 101 | 40 | 9 | 6 | 2 |
| 128 | K3 | 110 | 109 | 109 | 110 | 9 | 2 | 0 | 0 |
| 128 | K4 | 107 | 109 | 108 | 107 | 1 | 1 | 0 | 0 |

For confirmation K2, N/U/restored-U/A-only correct counts are 103/98/105/101. U gives 4 rescues/9 damages; restored-U gives 6/4 relative to N. Among 40 natural corrections,9 are blocked and 6 recover;3 remain wrong. One additional native-correct question is damaged by restoration. The all-four-control tie-free cases are Mercury_SC_400986 and TIMSS_2011_8_pg100. This is controlled conditional mediation evidence, not a percentage of all recurrent reasoning explained.

| Cohort | Receiver | Total meanwrong | Direct at native A | Restore effect | Restore 95% interval |
|---|---:|---:|---:|---:|---|
| 299 | K2 | -1.453247 | -1.375505 | +0.077742 | [+0.0375, +0.1203] |
| 299 | K3 | -0.467740 | -0.569363 | -0.101623 | [-0.1303, -0.0708] |
| 299 | K4 | -0.067917 | -0.115646 | -0.047729 | [-0.0628, -0.0317] |
| 128 | K2 | -1.394368 | -1.353923 | +0.040446 | [-0.0160, +0.0970] |
| 128 | K3 | -0.363932 | -0.506673 | -0.142741 | [-0.1813, -0.1032] |
| 128 | K4 | -0.032471 | -0.075399 | -0.042928 | [-0.0618, -0.0234] |

All intervals use10,000 paired empirical question bootstrap resamples, seed 20261112, separately within each cohort. They measure question-resampling variability, not training-seed replication, globally blind testing or multiplicity-corrected discovery. ConfirmationK2 total/direct meanwrong intervals are[-1.6180,−1.1792]/[-1.5816,−1.1351]. Restore meanwrong interval spans zero[-0.0160,+0.0970], whereas dynamic-margin interval is[+0.0020,+0.1431]. All four metrics and intervals remain in ROOT_CONFIRMATION128_QUESTION_BOOTSTRAP.json; no preferred metric determines disclosure.

## E5: late computation and diminishing gains

Historical Test conditional rescue rates decline69.23%→36.99%→19.19%, while rescues/damages remain378/51,81/34,33/13. Fewer remaining errors do not arithmetically explain the entire drop, but the risk sets change. New299 risk sets contain104/20/7 natural rescues; confirmation 128 contains 40/9/1. Its native69/103/110/107 trajectory includes a real K4 decrease.

Late R rollback has weaker mean effects, but restoring native attention has negative mean effects at K3/K4 in both cohorts. Confirmation late natural corrections show0/0 restored cases at K3/K4, compared with 4/1 in299; this limits late-path generalization. Native state changes remain nonzero. Mean R perturbation norms in299 are 4.3055/3.2560/2.5072; in128,4.2490/3.1886/2.4887. Corresponding actual-attention perturbations also change. These are varying natural-update interventions, not equal-amplitude sensitivity tests or proof that computation has converged.

Figures: figures/validation299_state_mediation.png; validation299_state_mediation_depth.png; confirmation 128_state_mediation.png; confirmation 128_state_mediation_depth.png; confirmation 128_query_factorial.png (each alsoPDF/SVG). The original cross-loop KV and Query figures remain available. Figure captions preserve risk-set and perturbation-magnitude limits.

## Evidence hierarchy and peer integration

Descriptive observations are full-cohort behavior/geometry; decodability is peer frozen-probe evidence; local causal effects are transplantation and receiver-conditioned rollback; natural-path evidence is subset-specific same-question block/restoration; the mechanistic explanation is the limited R-to-readout computation, not a complete upstream derivation.

Peer E3 region rollback and the completed64-question coarse Q/R producer experiments are substantive executed evidence, not merely technical pilots. Their independently audited bare replays pass, but same-cut matched controls have not yet established an attention/MLP writer. Cut0/8/16 effects and target perturbations are not dose-matched; cut 17 is a target anchor. Background-dependent Key-edge and whole-carrier effects cannot be equated. The peer64 overlaps this task's64 in only 9 questions, on different hardware/queues. The next17-condition writer-discrimination order is only PENDING and contributes no effect evidence. Our E4/E5 scope closes without waiting for it.

## Numerical, control and resource audit

Production uses BF16 model arithmetic and FP32 log-softmax, with raw candidate logits, exact top ties and same-hardware native baselines preserved. The pilot scoped FP32-SDPA/BF16-writeback diagnostics changed one of 48 KV predictions and one of 8 Query predictions; pilot R changed0/8. Late299 R changed0/8 at each depth, and confirmation 128 Query/R2/R3/R4 each changed0/8. These small diagnostics are not full-model FP32 guarantees. Two Validation mediator-control vectors differed by one BF16 element under cross-CPU FP32 recomputation; independent FP64 rounding bounds and exact same-environment recomputation passed. Identity and intended four-corner tensor bridges remain exact.

Historical/current native predictions differ on 7 pilot and 11 Validation IDs; new paired effects use current baselines. The original pilot donor cycle followed its pre-frozen selection-hash order, with 32/64 same historical-group donors. Formal frozen-map compliance holds, but the initial draft description of ID sorting was inaccurate and the donor mapping was stratum-associated. Later maps are explicitly ID-sorted. Auxiliary LF-versus-CRLF receipt hashes were corrected with preliminary records retained; runtime raw-byte identity was always recorded correctly. Neither correction changes raw outcomes.

Across this task's23 COMPLETE namespaces:491 unique questions,28,722 actual forward windows,5,140.79s summed collector time and 5,339.49s summed UTC command windows. Recorded-forward-window peak allocated/reserved memory was 3,209,819,136 /4,441,767,936 bytes; process CPU maximum RSS was 3,581,744KiB. These sums are not total elapsed research time, and recorded-window GPU peaks exclude model loading and between-forward derivation/serialization. Historical2590 metadata reanalysis is not2590 new causal experiments; new Test interventions are zero. Technical repeats and reused questions are not independent samples.

## Final limits and contribution

The earlier attention/MLP component that generated the useful R or KV state is still TARGET_UNRESOLVED. One-site mixed-state interventions identify a controlled local path, not the complete origin of the second-loop benefit. Unrelated controls cover one frozen direction per question, not every off-manifold explanation. No mediated percentage, skipped-loop shortcut, pure-MLP attribution, semantic information-origin claim or global hidden-state convergence follows.

Activation patching and mediator analysis are established methods; see [Vig et al.](https://arxiv.org/abs/2004.12265) and [Zhang and Nanda](https://arxiv.org/abs/2309.16042). The contribution here is the specific audited Ouro readout finding and its failures, not a new causal methodology, broad priority claim or publication-readiness verdict. The complete second-loop mechanism remains unsupported until an actual upstream writer is localized with matched controls and its path independently tested.

## Reproducible evidence index

The six required E4 reports, three E5 reports, scientific reviews, protocols/orders, REPRODUCE.md and SOL_EXECUTION_LEDGER.md describe execution and claims. ROOT_CONFIRMATION128_ALL_METRICS.json independently recomputes5,152 endpoint rows and 208 four-metric/natural-group summaries; it includes meanwrong, recipient-fixedwrong, global-K1-fixedwrong, dynamic margin, correctness and top ties. ROOT_CONFIRMATION128_DECISION_CASES.json preserves KV/Query cases; the R2/R3/R4 review JSONs preserve every four-corner contrast and damaged case. ROOT_LATE3/4_REVIEW.json and both QUESTION_BOOTSTRAP.json files support the late-depth/uncertainty tables. Compact raw tensors, scores, controls, logs, source hashes and execution identities remain in their original result namespaces. No omitted unfavorable cell is replaced by a selected example.
