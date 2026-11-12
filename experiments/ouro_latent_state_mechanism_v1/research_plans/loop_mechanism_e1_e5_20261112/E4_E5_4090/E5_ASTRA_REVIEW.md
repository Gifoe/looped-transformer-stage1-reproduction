# Final Astra E5 scientific review

Final status, 2026-11-13: all fixed E4/E5 experiments through Orders 005/006 are complete and independently audited. No further GPU intervention is authorized by this review. The remaining generating-component uncertainty is an explicit scientific limit, not unfinished E4/E5 execution.

The same controlled R-state/actual-attention operator was measured at K2/K3/K4 on 299 and independently selected128. The result is a depth-dependent local computation, not a demonstrated single algorithm that simply saturates.

| Cohort | Receiver | Native correct | U correct | Restored-U correct | A-only correct | Natural rescues | Lost under U | Recovered by A | Strict four-control recoveries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 299 | K2 | 243 | 220 | 240 | 235 | 104 | 29 | 22 | 11 |
| 299 | K3 | 258 | 251 | 256 | 253 | 20 | 9 | 4 | 2 |
| 299 | K4 | 261 | 256 | 256 | 263 | 7 | 5 | 1 | 1 |
| 128 | K2 | 103 | 98 | 105 | 101 | 40 | 9 | 6 | 2 |
| 128 | K3 | 110 | 109 | 109 | 110 | 9 | 2 | 0 | 0 |
| 128 | K4 | 107 | 109 | 108 | 107 | 1 | 1 | 0 | 0 |

| Cohort | Receiver | Total meanwrong | Direct at native A | Restore effect | Restore 95% interval |
|---|---:|---:|---:|---:|---|
| 299 | K2 | -1.453247 | -1.375505 | +0.077742 | [+0.0375, +0.1203] |
| 299 | K3 | -0.467740 | -0.569363 | -0.101623 | [-0.1303, -0.0708] |
| 299 | K4 | -0.067917 | -0.115646 | -0.047729 | [-0.0628, -0.0317] |
| 128 | K2 | -1.394368 | -1.353923 | +0.040446 | [-0.0160, +0.0970] |
| 128 | K3 | -0.363932 | -0.506673 | -0.142741 | [-0.1813, -0.1032] |
| 128 | K4 | -0.032471 | -0.075399 | -0.042928 | [-0.0618, -0.0234] |

K3/K4 restoration mean effects are negative in both cohorts, despite some299 decision recoveries. Confirmation128 has no late natural-correction recovery and native K4 loses 3 correct answers. This is a required counterexample to monotone improvement and universal late-path restoration. Small changing risk sets and shrinking natural perturbations limit inference; no equal-size sensitivity, hidden-state convergence or original upstream writer is established. Historical complete2590 trajectories remain descriptive evidence, separate from 491 new unique intervention questions.

All E5 fixed execution is complete; the unresolved writer is the scientific boundary. Detailed states, controls, four metrics, bootstrap intervals and raw cases are linked in E5_MECHANISM_BY_DEPTH.md, E5_CAUSAL_DEPTH_COMPARISON.md and FINAL_RESEARCH_REPORT.md.
