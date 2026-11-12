# Final mechanistic model

Final status, 2026-11-13: all fixed E4/E5 experiments through Orders 005/006 are complete and independently audited. No further GPU intervention is authorized by this review. The remaining generating-component uncertainty is an explicit scientific limit, not unfinished E4/E5 execution.

The supported explanation is a **local, receiver-dependent state-to-readout mechanism**, not a complete derivation of the K1→K2 gain. By L18, the recurrent R residual has changed. Changing that residual modifies both its actual attention computation and the residual/suffix computation that remains when native attention is held fixed. Exact native-attention restoration recovers some natural corrections lost by the residual rollback. The strength and sign of these effects depend on the receiving recurrent state and question.

## What the evidence establishes

1. **Descriptive observation:** historical Test correct counts 626/953/1000/1020 coexist with late rescues and damage; later states keep changing. Current299 native counts 148/243/258/261 and confirmation 128 counts 69/103/110/107 are separate hardware-cohort observations. K4 worsens confirmation accuracy by 3; monotone improvement is false there.
2. **Decodability:** peer frozen probes read more answer-relevant structure at K2, including under a fixed K1-trained decoder. This does not establish information creation or causal use of a probe direction.
3. **Local causal effect:** later KV transplanted into K1 improves many decisions, but forward restoration and natural-K2 rollback usually affect different questions. Natural Query direction has a subset-specific effect. These are conditional readout effects, not a single transferable necessary-and-sufficient component.
4. **Natural-path evidence:** same-question R rollback in a natural receiver computes actual A(U); restoring native A while retaining the rollback tests a distinct downstream mediator. Confirmation128 contains two tie-free natural-correction recoveries specific against all four controls: Mercury_SC_400986 and TIMSS_2011_8_pg100.
5. **Mechanistic explanation with a boundary:** the local R→attention→suffix route is causally supported for a subset, and a substantial continuous effect bypasses this L18 attention mediator. The bypass includes later attention, normalization, residual and MLP computation; it is not identified as pure MLP. The upstream writer that produced R remains unresolved.

## Controlled mechanism and its counterexamples

N=(X_current,A_current); U=(X_previous,A(U)); restored-U=(X_previous,A_current); A-only=(X_current,A(U)). A(U) is actually computed under the state block, not copied from the preceding loop. R is the final token; non-R x/Q/K/V/SDPA hashes and the unchanged full suffix are verified. Native mediator restoration preserves the actual blocked-state QKV and exactly restores the normalized attention increment.

| Cohort | Receiver | Native correct | U correct | Restored-U correct | A-only correct | Natural rescues | Lost under U | Recovered by A | Strict four-control recoveries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 299 | K2 | 243 | 220 | 240 | 235 | 104 | 29 | 22 | 11 |
| 299 | K3 | 258 | 251 | 256 | 253 | 20 | 9 | 4 | 2 |
| 299 | K4 | 261 | 256 | 256 | 263 | 7 | 5 | 1 | 1 |
| 128 | K2 | 103 | 98 | 105 | 101 | 40 | 9 | 6 | 2 |
| 128 | K3 | 110 | 109 | 109 | 110 | 9 | 2 | 0 | 0 |
| 128 | K4 | 107 | 109 | 108 | 107 | 1 | 1 | 0 | 0 |

Strict counts exclude adjacent-native/block/restored endpoint ties, require both mediator controls to fail to recover, and both state controls to preserve correctness. They are finite-cohort case classifications, not explained fractions. In299/K2,15 meet the mediator-only strict criterion and 11 also pass state controls. In confirmation/K2,6 of 9 lost natural corrections recover but only 2 meet all-control specificity. The same criterion yields no late-depth recovery in confirmation 128.

| Cohort | Receiver | Total meanwrong | Direct at native A | Restore effect | Restore 95% interval |
|---|---:|---:|---:|---:|---|
| 299 | K2 | -1.453247 | -1.375505 | +0.077742 | [+0.0375, +0.1203] |
| 299 | K3 | -0.467740 | -0.569363 | -0.101623 | [-0.1303, -0.0708] |
| 299 | K4 | -0.067917 | -0.115646 | -0.047729 | [-0.0628, -0.0317] |
| 128 | K2 | -1.394368 | -1.353923 | +0.040446 | [-0.0160, +0.0970] |
| 128 | K3 | -0.363932 | -0.506673 | -0.142741 | [-0.1813, -0.1032] |
| 128 | K4 | -0.032471 | -0.075399 | -0.042928 | [-0.0618, -0.0234] |

The confirmation K2 direct effect is−1.353923 with question-bootstrap95% interval[-1.5816,−1.1351]. Its restore meanwrong effect is+0.040446 with interval[-0.0160,+0.0970]; a general positive mean effect is less securely supported than in299. Dynamic-margin restoration is+0.072754 with interval[+0.0020,+0.1431]. These metric differences must remain visible. Restored-U reaches105 correct versus native103 while retaining a large negative meanwrong effect: accuracy and continuous competition are different outcomes.

Late restoration has a negative average gap effect in both cohorts, even though some299 questions recover. It is not the same positive early operation merely becoming smaller. Natural residual and attention perturbations also shrink, so the measurements do not isolate equal-size receiver sensitivity. Small late risk sets (confirmation 9 and 1 natural rescues) prevent strong generalizations about late correction mechanisms.

## What did not reproduce

Confirmation KV rollback damages 3 questions versus 5 under unrelated control. Two natural corrections are lost, one survives both controls but has a relevant tie: strict KV necessity count is 0, versus 9 in299. This does not strongly reproduce the299 strict decision result. Query-only and jointQKV each retain 3 strict natural-correction losses in128. Continuous KV effects and discrete specificity must not be conflated.

The earlier attention/MLP component that generated the useful R or KV state is still TARGET_UNRESOLVED. One-site mixed-state interventions identify a controlled local path, not the complete origin of the second-loop benefit. Unrelated controls cover one frozen direction per question, not every off-manifold explanation. No mediated percentage, skipped-loop shortcut, pure-MLP attribution, semantic information-origin claim or global hidden-state convergence follows.

The original KV299 followed Train 64 discovery. Later Q/R operators and the late-depth extension were selected after viewing299, so those results are exposed-cohort analyses. Order 006 froze a label-blind remaining-Train 128 by ID hash before its new interventions, excluding25 demonstrations, both discovery64 sets and token-identical inputs in the prior64/299/peer64. There were 975 eligible IDs and 144 excluded Train IDs. The128 questions were not used to choose this task's new causal operators, but had historical full-cohort/eight-corner exposure and may have entered peer probes. This is not a globally unseen or blind test. Cohorts are never pooled.

See FINAL_RESEARCH_REPORT.md, E4_MEDIATION_ANALYSIS.md, E5_CAUSAL_DEPTH_COMPARISON.md and MECHANISM_COUNTEREXAMPLES.md for complete evidence and failure cases.
