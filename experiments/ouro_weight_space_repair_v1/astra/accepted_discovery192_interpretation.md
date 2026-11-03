# Accepted mechanism checkpoint: original-call discovery, 192 questions

Status: accepted discovery evidence only, independently checked from local artifacts on 2026-11-03. The formal layer screen and shared confirmation have now completed; the independent stop/go decision is recorded in `independent_stop_go_after_shared.md`. This note preserves the discovery-stage reasoning; its pending-screen language below is historical. No directional or weight repair is established. No validation/test data were consulted.

## Evidence provenance

- `results/discovery_original/COMPLETE_192.json`: complete, 192 questions; implementation SHA256 `f3678de0a5fee28e39cf13c03292e12a5db544478c47b84c9f5822fc8c2ef6f9`.
- `results/original_call_parity/COMPLETE.json`: exact original-call parity, original-repeat exactness, and exact separate/intermediate depth parity all true on the audited non-test inputs. This establishes the checked cases, not an empirical guarantee for every possible input.
- `results/discovery_original/ANALYSIS.json`: source of the aggregate and representative results below.
- `results/discovery_original/PARITY.json`: HFLM K4 prediction agreement; maximum reported per-token discrepancy 0.002395027562191612. This separate harness check is tolerance-based. Do not confuse it with zero-error parity against the exact old direct model call.

The accepted execution uses `model.model(input_ids=x, use_cache=False)` with original candidate scoring. Explicit all-one-mask and cached runs are excluded. Fixed global 25-shot examples were explicitly authorized, so prompt comparability applies to the rerun methods, not to historical varying-demo outputs.

## Accepted discovery facts

| Depth | Correct / 192 | Accuracy | Mean normalized gold margin |
|---|---:|---:|---:|
| K3 | 119 | 61.98% | 0.063078 |
| K4 | 118 | 61.46% | 0.063375 |
| K5 | 120 | 62.50% | 0.052379 |
| K6 | 118 | 61.46% | 0.055322 |

K4-to-K5 produces **3 harms and 5 corrections**, with 115 stable-correct and 69 stable-wrong questions. All three harms exceed the declared margin epsilon. Two of the three harms recover correctness at one or more tested interior normalized-chord points. **Zero harmful cases meet the declared sampled-path overshoot criterion.** The remaining harm is unrecovered on this grid.

There is no observed fifth-loop aggregate accuracy loss: K5 gains two correct answers over K4. K6 equals K4 and is two answers below K5. These small cohort differences do not establish a universal depth trend. Mean margin decreases at K5 despite higher accuracy; task accuracy and mean answer support measure different aspects of behavior.

## What follows scientifically

The two interpolation recoveries show path/readout sensitivity. They do **not** demonstrate a locally useful fifth-loop direction that overshoots: the stricter prespecified sampled-path criterion is not met among harms. The correct formulation is “interpolation-recoverable harms,” not “two confirmed overshoots” or “H1 established.” Likewise, absence of sampled overshoot does not falsify H1 outside the tested path/grid.

The sole grid-unrecovered harm does not establish H2. Untested smaller step scales, alternative state geometry, and readout sensitivity remain possible. H3 is also unproven: accessing an earlier state or mixed state is not proof that the unmodified final state retains all needed information or that its head is defective.

Only three harmful transitions exist in this discovery cohort. Selecting and screening failures can uncover a local mechanism, but cannot support broad claims about recurrent degradation. The necessary standard for continuing is component-specific evidence that transfers beyond these selected questions, not merely a best intervention among many trials.

## Current representative cases and pending screen

The two frozen harmful representatives are:

| Case | K4 margin | K5 margin | Best tested interior margin | Interpretation |
|---|---:|---:|---:|---|
| MCAS_2013_5_29397 | 0.001446 | -0.006938 | 0.000702 | Correctness recovered, but interior margin stays below K4; no sampled overshoot |
| Mercury_7180758 | 0.001562 | -0.017957 | -0.001610 | Sole grid-unrecovered harm; mechanism unresolved |

The other six representatives are two native rescues, two stable-correct and two stable-wrong questions. The sole unrecovered harm is **already included**. Earlier concerns about a second omitted unrecovered harm arose under the superseded mask path and no longer apply.

The formal screen uses the unchanged accepted kernel and must pass exact recording-only parity. Its fifth-loop boundary controls should also reproduce the corresponding discovery endpoint chord scores exactly at K5; downstream K6 changes then isolate effects of feeding the damped boundary into further recurrence. Raw/post-branch contributions, and alignment to the actual boundary displacement, are descriptive statistics until a controlled intervention confirms their role.

No intermediate screen winners should drive a change in cases, strengths or hypotheses before the frozen screen is complete. Forty-eight post-norm branch damping trials on eight selected cases create substantial selection risk. Module ranking remains exploratory, even if the score differences are exact and deterministic.

## Independent stop/go rule after completion

**Current decision: no PCA search and no weight edit justified.** Wait for the completed formal screen.

1. Stop the current H2 search if no selective branch effect exceeds the tested boundary-control benefit, or if rescues are offset by regressions on representative native successes. State “no supported shared harmful component found,” not “H2 is impossible.”
2. If a narrow module lead remains, confirm at most two modules under all-loop post-norm damping before expanding. This checks whether the loop-local effect survives shared invocation. The current cohort has no omitted unrecovered case requiring automatic addition.
3. All-loop scalar damping failure does not logically falsify all raw projection directions. It does reduce the justification for spending further budget on this particular lead. A new direction experiment would require independent component-specific evidence, not a desire to obtain a positive edit.
4. Only a reproducible lead beyond scalar/boundary controls warrants one or two fixed, label-blind direction diagnostics. Require pre-branch-norm interventions, norm-matched/random controls, untouched train-selection transfer, and an equivalent actual BF16 shared-weight intervention before claiming H2 or training-free repair.

The final report must separate discovery, selected-case localization, held-out transfer and final frozen evaluation. No current accepted result demonstrates a transferable harmful direction or a publishable repair method.

## Quarantine

Earlier 64/192-question interpretation notes were moved to `astra/superseded_explicit_ones/` and retain prominent superseded banners. Their counts, case categories and overshoot labels must not be merged into this accepted analysis. Their numerical differences are engineering evidence that changing the attention-mask dispatch materially changed the diagnosis, not additional scientific samples.
