# First complete ARC-Challenge evaluation

2026-10-30. **Fresh inference on released final nl33w; not nl23 QA checkpoint reproduction.**

The entire official four-choice ARC-Challenge test (1,165 of1,172 raw rows) was scored under ON and CUT, with the exact same released adapters/backbones. It completed in25min40sec, including loading and first-batch scoring parity checks. Raw letter logits and probabilities for both receivers are retained in146 batch files. All original16631/595/3165 train/validation/test benchmark bytes have also been reconstructed: SHA-256 exactly matches the author's saved data hash.

| Receiver | CUT | ON | Corrected | Damaged | Net |
|---|---:|---:|---:|---:|---:|
| Quill | 1010/1165 =86.70% | 1003/1165 =86.09% | 19 | 26 | -7 |
| Ember | 967/1165 =83.00% | 987/1165 =84.72% | 42 | 22 | +20 |
| Mean of both probabilities | 1024/1165 =87.90% | 1021/1165 =87.64% | See raw probabilities | See raw probabilities | -3 |

Quill ON-minus-CUT is-0.6009pp (exact McNemar p=.3713; paired bootstrap95% CI[-1.7167,+0.5150]pp). Ember is+1.7167pp (p=.01686; paired bootstrap95% CI[+0.3433,+3.0901]pp). The author's more conservative difference intervals include zero for both; interval construction and exact paired test are separate procedures. Holm across these two receiver tests gives Ember p=.03372 and Quill p=.3713. These condition on a single released training run, not training-seed uncertainty.

CUT strata: both correct902, only Quill correct108, only Ember correct65, both wrong90. Within the108 only-Quill-correct cases, Quill is damaged11/108 and Ember rescued36/108. Within the65 only-Ember-correct cases, Quill is rescued10/65 and Ember damaged8/65. Among those11 Quill damage cases,7 adopt Ember's CUT wrong letter; among the8 Ember damage cases,5 adopt Quill's CUT wrong letter. These are descriptive answer-agreement signatures. They do not establish causally that the sender's wrong judgment was copied.

## Relation to the requested official nl23 analysis

The official nl23 saved records on the same1,165 ARC items have CUT strata964/83/50/68. The83 and50 counts were independently verified. Quill is damaged10/83 where only Quill was correct, while Ember is rescued64/83. Where only Ember was correct, Quill is rescued15/50 and Ember damaged29/50. These values are **published-record reanalysis**, available in published_arc_nl23_disagreement.csv. Saved nl23 JSON lacks actual predicted letters/probabilities.

The strict nl23 baseline remains unfulfilled, and the user has explicitly deferred full nl20→nl23 reconstruction. It is no longer a prerequisite for the current final-nl33w causal audit. Existing training preparation is preserved. No nl33w artifact is relabeled as nl23.

## Interpretation

The final checkpoint exhibits asymmetric recipient effects and many cancelled rescue/damage transitions. Communication does not improve the simple probability mean in this run. These observations neither identify a hidden-state bottleneck nor establish any need for recurrence. Negative transfer and asymmetry were already known; no novelty claim follows from this first evaluation.

The current nl33w mechanism audit is complete: see REPORT.md and CASE_STUDIES.md for the95-question original-shape8-layer study and16-target validation. The new content-specific repair criterion failed; Loop and full retraining are not justified by these data.
