# E7 scientific director review

2026-11-14. Astra reviewed the frozen E7 order and the completed Sol-produced `analysis/E7_CAUSAL_REVIEW_VERDICT.json`, `TRAIN_REPORT.md`, `VALIDATION_REPORT.md` and `ROOT_INDEPENDENT_ARITHMETIC_AUDIT.json`. Astra did not run analysis, scripts or GPU inference. C1's frozen hypotheses, samples, operators and controls are unchanged.

**Decision: inconclusive for replicated mean loop-gain dependence; no evidence of practical invariance.** The full22,280 views cover1,393 questions, not22,280 independent samples. Primary four-choice strata are Train1,092 and Validation295; the six three/five-choice questions remain separate.

The following effects concern the change in semantic correctness from K1 to K2, averaged within question over the frozen offset grid. They are not direct improvements in K2 accuracy.

| Contrast | Train estimate pp [family-adjusted98.3333% CI] | Validation estimate pp [95% CI] |
|---|---|---|
| Position P | +1.0302 [−1.3507,+3.3196] | +0.4237 [−2.9096,+3.7860] |
| Symbol S | +2.0986 [+0.3764,+3.7775] | +0.0847 [−2.7401,+2.8531] |
| P×S interaction | −4.3244 [−8.6625,−0.0407] | −3.6535 [−10.3955,+3.1262] |

S and the interaction meet the Train interval criterion, but no contrast meets the prespecified same-direction Validation replication criterion. The Validation interaction estimate is compatible with a negative effect and with zero; it is not evidence of a reversal. None satisfies the frozen±1pp practical-equivalence criterion. Thus neither “the gain is protocol-invariant” nor “replicated protocol dependence explains the gain” is supported. These are question-bootstrap intervals on historically exposed ARC items, not model-seed replication or independent benchmark confirmation.

The independent arithmetic audit passes raw-unit checksums, mapping, first ties, per-depth predictions, complete-grid coverage, exact question-block primary means and fixed-pair correctness. Its scope is not a second GPU execution or an independent bootstrap replication. The reported intervals retain Sol's original analysis provenance.

## Constraint on mechanism interpretation

E7 tests whole natural inputs. P changes the causal context of options; S changes structural answer symbols in demonstrations and the query. Consequently, even the Train effects do not isolate label bias, a writer, information creation, or an attention/MLP component. The p=s diagonal preserves query candidate symbols but changes demonstration symbols; it is not a pure whole-prompt position intervention. Aggregate gain equivalence, had it passed, would still not prove identical internal computations or absence of question-level effects.

C1 remains a conditional internal-path test under its original fixed prompt. E7 neither invalidates that estimand nor grants it semantic or reencoding invariance. A supported C1 branch would explain part of the L10–L16 N−P intervention's effect through L18, not prove that it generates new answer information or accounts for K1→K2 correction across representations. The true full-sequence recurrent-boundary-to-Q9/update-generation edge remains unresolved. E7 did not intervene on that boundary and cannot close it. No C1 condition is added, dropped or reselected from these E7 outcomes.

## Closed practical branches

The unique frozen two-K2-view pair fails: Train936/1,092 versus nativeK4 964, and Validation244/295 versus257. Its paired accuracy-difference intervals versusK4 are negative in both splits: [−4.3956,−0.7326]pp and [−7.7966,−1.0169]pp. It also fails against the frozen strong output-only comparator. This does not discredit the natural-input measurements, but it supplies no method improvement.

The previously selected gate threshold0.25 is an offline readout result. Every captured example still executed four bodies; estimated prefix depth is not measured halting cost. The user's newer mechanism-first instruction supersedes prior profiling permissions: actual pair/gate profiling was not run, and neither is authorized by this review. Historical report wording about eligibility is not current execution authority.

Publish the complete E7 evidence and its negative/inconclusive findings. Continue the already frozen C1 causal experiment; do not turn this review into a new optimization search. The overall mechanistic research goal remains open.
