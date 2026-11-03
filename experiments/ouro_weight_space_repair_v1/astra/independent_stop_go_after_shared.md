# Independent Astra STOP / GO review after formal shared confirmation

**Decision: STOP the current H2-to-weight-repair route. Do not start PCA or materialize a weight edit from these results.** This is a bounded decision about the screened leads and the stated experiment, not a claim that H2 is universally false or that every projection intervention would fail.

I made this decision after inspecting completed original-call screen and shared-confirmation results, not from the prepared no-edit reporting pipeline. No selection, validation or test evidence was accessed. No new GPU job was run.

## Evidence integrity checked

Read `layer_screen_original/MECHANISM_ANALYSIS.json`, its aggregate CSV and candidate-score records, no-op gate and boundary identity receipt; and all five supplied shared-confirmation files (`COMPLETE`, `FROZEN`, `NOOP_PARITY`, `GLOBAL_VS_LOOP5`, `scores`). Recomputed outcome counts from candidate records against `discovery_original/scores.jsonl`.

- Screen: 408 case-condition records, eight selected discovery questions, 48 physical modules. Six modules met the local screen criterion before the permitted two-module shortlist was chosen.
- Shared confirmation: 32 records, exactly two modules, two invocation scopes, eight questions. No omitted unresolved harm needed adding: the sole grid-unrecovered harm was already present.
- Recording no-op checks passed; all 16 boundary-step/case K5 candidate-score identities had exactly zero error against accepted discovery chords. Shared no-op K3/K4/K5/K6 errors were zero.
- Reused loop5 K4/K5/K6 method records exactly match the original completed screen. Independently checked all 64 global-minus-local contrast records against the underlying scores.

## Actual depth tradeoffs on the fixed eight representatives

Entries are correct answers out of eight. These selected cases are not a population accuracy estimate.

| Intervention | K3 | K4 | K5 | K6 |
|---|---:|---:|---:|---:|
| Native | 4 | 4 | 4 | 6 |
| MLP physical layer 0, loop5-only damping .5 | 4 | 4 | 5 | 4 |
| MLP physical layer 0, all-loop damping .5 | 5 | 4 | 4 | 4 |
| Attention physical layer 10, loop5-only damping .5 | 4 | 4 | 5 | 5 |
| Attention physical layer 10, all-loop damping .5 | 2 | 2 | 2 | 4 |

Both shortlisted local interventions beat the tested boundary controls at K5: they achieve one correction with no regressions, whereas boundary step .25 has one correction/two regressions and step .5 has zero corrections/one regression. This is real module-specific causal sensitivity, not merely a correlation in output norms. It is sufficient to justify the shared-invocation check that was performed.

The benefit does not survive that check as a net K5 repair:

- **MLP layer 0:** all-loop K5 has one correction and one regression, net zero. K4 also swaps one correction for one regression. K6 has zero corrections and two regressions. Mean margin changes at K3/K4/K5 are -0.06347/-0.04081/-0.00733; K6's positive mean margin change (+0.01349) coexists with two fewer correct answers. Choosing that mean alone would hide the failure on the target metric.
- **Attention layer 10:** all-loop application has zero corrections and two regressions at each of K3/K4/K5/K6. Its K5 local rescue disappears, and the two representative native 4-to-5 rescues become incorrect. This lead fails the shared-invocation prerequisite directly.
- Even loop5-only damping fails to stabilize deeper behavior: MLP loses two native K6 successes and attention loses one. K5 improvement cannot be presented as improved extra-loop scaling.

## Strongest argument for continuing, and why it does not pass the gate

The MLP result is not uniformly negative. The unique grid-unrecovered harm, `Mercury_7180758`, is rescued by local MLP damping and remains rescued when the damping operates at every loop. Its all-loop K5 gold margin rises from -0.01796 to +0.11671. The intervention also makes that same question correct at K3 and preserves correctness at K4/K6. This is a substantive single-question causal lead and should be reported rather than obscured by aggregate counts.

However, the same all-loop manipulation destroys the native K5 rescue on `Mercury_SC_415737`, and harms `MCAS_2013_5_29397` at K4/K6. The lead therefore demonstrates a tradeoff between useful and harmful effects of a broad branch, not an isolated removable component. Only one originally unrecovered question supplies the favorable pattern, after a 48-module search on eight selected cases. The unchanged train-selection set has not been used; there is no unseen-question transfer evidence.

A more selective projection might separate those effects. Broad damping failure cannot logically exclude that possibility. But the current evidence does not identify the separating direction: there is no validated non-collinear component, no direction-specific norm/random comparison, and no repeated harmful-direction pattern across independent questions. Label-blind PCA would identify high-variance output structure; it would not, by itself, identify harmful computation. Searching PCs now would introduce a new representation search after the predeclared shared-invocation gate failed. The remaining GPU budget is not evidence for that escalation.

## Scope of the conclusion

The predeclared decision rule requires a localized lead to retain useful all-loop effects without losing the representative native successes before expanding into directional diagnosis. Neither shortlisted lead satisfies it. Moving to lower-ranked modules, weaker strengths or many principal components after these outcomes would widen the search beyond the bounded mechanism-first plan.

The strongest supported conclusion is:

> Selected fifth-loop errors are sensitive to specific branch contributions, and one grid-unrecovered error can be rescued by damping an early MLP. The screened effects are depth- and question-dependent: applying the same suppression across shared invocations yields compensating regressions and fails to improve K5/K6 behavior. These results do not establish a transferable harmful weight-space component and do not justify a weight-edit candidate in the present run.

Do not replace that conclusion with “H1 explains everything,” “readout failure is proven,” “PCA was tested and failed,” or “training-free repair is impossible.” Directional PCA, held-out directional transfer and actual weight editing were **not attempted because the progression gate failed**. The accepted discovery evidence still contains zero harmful cases satisfying the declared sampled overshoot criterion; it does not support a universal H1 explanation either.

## Recommended completion

Freeze `primary_weight_edit = none` with the stop reason and tested conditions. Preserve exact candidate scores, depth curves and all six locally qualifying modules. Any authorized frozen baseline/readout evaluation may proceed under its separately fixed protocol, but it cannot retrospectively change this direction-search decision. No further selection data are needed to decide whether these two screened broad-suppression leads meet their existing gate.

Reopening the research direction should be a separately scoped experiment with a new preregistered criterion and evidence requirement, such as independent recurrent failures sharing a candidate direction. It should not be represented as successful continuation of this failed shared-repair gate.
