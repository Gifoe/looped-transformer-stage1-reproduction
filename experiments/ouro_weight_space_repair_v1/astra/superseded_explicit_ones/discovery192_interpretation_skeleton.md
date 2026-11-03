# Mechanism interpretation checkpoint: frozen 192-question discovery

**SUPERSEDED ENGINEERING DIAGNOSTIC — NOT FORMAL MECHANISM EVIDENCE.** The explicit-all-ones attention-mask path differed from the exact original no-mask invocation. Every number and selected failure below is provisional old-mask output retained for engineering provenance only. Formal conclusions, representative selection, and causal screening must be regenerated under `discovery_original`; this skeleton must not supply numerical claims to the final report.

Verified from this experiment's `results/discovery_audit/ANALYSIS.json` on 2026-11-03. These are official-training discovery outcomes, with fixed global 25 demonstrations excluded from the discovery pool. No validation or test outcomes were consulted.

| Depth | Correct / 192 | Accuracy |
|---|---:|---:|
| K3 | 120 | 62.50% |
| K4 | 118 | 61.46% |
| K5 | 118 | 61.46% |
| K6 | 116 | 60.42% |

K4-to-K5 changes six correct answers to incorrect and six incorrect answers to correct. Five of the six harms exceed the declared margin threshold. Four harms recover correctness somewhere on the fixed interior normalized-chord grid; two harms satisfy the stronger sampled-path overshoot criterion. Two harms remain unrecovered on that bounded grid.

**Established:** individual loop-induced harms coexist with equally many corrections. Fifth-loop aggregate accuracy does not decline. K6 has two fewer correct answers than K4, a small discovery-cohort observation that must not be inflated into a general failure law. Mean gold margins fall from approximately 0.0631 at K4 to 0.0519 at K5, so equal accuracy does not imply unchanged answer support.

**Not established:** unrecovered cases are not evidence of a shared harmful weight direction. The finite grid may miss useful step sizes; the normalized chord is not the only plausible geometric path; decision-boundary/readout effects can coexist with computation effects. Two sampled overshoots support H1 compatibility, not a universal H1 explanation. Four correctness recoveries alone do not prove a locally beneficial update direction.

## Pending causal screen: fill only after COMPLETE.json

- Exact no-op K4/K5/K6 parity: [runner evidence and analyzer summary].
- Loop5 boundary step .25/.5, downstream K6 effects: [corrections/regressions and margins on the same representatives].
- All 48 post-branch-norm scalar damp screens: [complete results, not only best layers].
- At most two physical-module leads: [exploratory rank and individual effects].
- All-loop application of those leads: [unmeasured until actually run].
- Early-loop regressions and effect on originally improved questions: [counts].

Current representatives include one unresolved harm (`Mercury_7215810`) and one interpolation-recovered, sampled-overshoot harm (`MCAS_2013_5_29397`). The second unrecovered harm is not part of the fixed eight-case screen. Do not silently replace representatives after seeing intervention outputs. If a limited lead survives, evaluate the omitted harm in a separately recorded follow-up with at most the shortlisted two modules; count this as additional discovery, not validation.

## Bounded decision branches

1. **Stop the current H2 search** if no branch yields a selective improvement beyond the tested boundary controls, or if apparent gains come with comparable representative regressions. Report unresolved individual failures and H1/H3-compatible sensitivity. Do not create an edited checkpoint merely to complete a deliverable list.
2. **Perform only a small shared-invocation confirmation** if a loop5-only branch improves the unresolved harm and selected-question outcomes but all-loop effects are unmeasured. Damping a branch after RMSNorm is a localization tool and may simply implement selective step-size reduction; it is not equivalent to shrinking its raw output weight.
3. **Allow one or two label-blind direction diagnostics** only when a localized lead remains useful beyond boundary controls and survives all-loop confirmation without destroying earlier computation. Use a frozen, small set of weight/PCA directions built from unlabeled discovery activation statistics. Evaluate direction removal, norm-preserving/step-size controls, and seeded matched random directions before any permanent weight edit. Labels may assess the fixed finite candidate list, not fit the direction.
4. **Reject the shared-component claim** if directions fail on untouched train-selection questions or their materialized BF16 shared-weight versions fail to reproduce activation benefits. Successful selected-case lesion results remain useful negative-mechanism evidence, but do not answer the transferable-repair question affirmatively.

The automated analyzer's shortlist is a descriptive resource-allocation aid. It cannot establish H2; it always records `h2_established=false`. Its all-loop gate is deliberately explicit: absent all-loop measurements means transfer is unmeasured, not disproved. No p-values from the 48-module representative-case search should be presented as confirmatory.

## Report wording to preserve

“On the frozen 192-question ARC training discovery cohort, the fifth loop changed answers in both directions without changing total accuracy. Most identified harms were recoverable on the tested interpolation path. We therefore tested whether the remaining failures contained a localized component beyond step-scale and readout sensitivity. [Insert completed causal result.] These observations [do/do not] justify a bounded shared-direction diagnostic; they do not by themselves establish a transferable weight-space repair.”

Preserve the unsuccessful cached-BF16 readiness comparison in implementation limitations. Accepted scientific scores use whole-candidate native no-cache inference; native K3/K4/K5 intermediate-depth invariance passed. Do not claim the cache optimization met numerical parity merely because its initial predictions matched.
