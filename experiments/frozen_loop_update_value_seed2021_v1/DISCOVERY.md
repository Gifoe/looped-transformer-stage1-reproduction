# 400-question discovery result

This is exploratory OpenBookQA training-split evidence, not final confirmation. No head training or final-task result is claimed here.

| Rule | Correct /400 | Accuracy | Net corrections vs Adaptive |
|---|---:|---:|---:|
| Adaptive LoopCD |192|48.00%|0|
| Original / Keep |193|48.25%|+1|
| Fixed LoopCD |189|47.25%|-3|
| Rollback to depth3 |193|48.25%|+1|
| Known-question loss proxy |191|47.75%|-1|
| Update alignment |188|47.00%|-4|
| Confidence selection |189|47.25%|-3|
| Token support |190|47.50%|-2|
| Two-format support |188|47.00%|-4|
| Two-format ensemble |180|45.00%|-12|
| Four-action oracle |216|54.00%|+24|

None of the predefined training-free selectors improves over Adaptive on this discovery sample. Lower known-question predictive loss does not show the expected signed benefit: prompt-residual AUC is0.404 for Keep,0.464 for Forward and0.380 for Rollback relative to Adaptive, with only31,11 and37 nonzero events respectively. These are small exploratory event sets; their signs are not inverted to construct another rule.

Rollback recovers11 errors made by Keep but loses11 Keep-correct answers. Any of depths1–3 recovers43 Keep errors. The four allowed actions still all fail on184 questions. Oracle recoverability is therefore real but insufficient evidence for an implementable selector.

Decision: proceed to the already bounded supervised test. Collect official-view trajectories for3200 separate training questions, then evaluate the fixed linear heads and unchanged training-free policies on800 separate validation questions. No extra model complexity or final-test tuning is authorized by these discovery outcomes.

The first run stopped after307 records on the single-token prompt `chlorophyll`, which cannot supply a causal known-question-token target without modifying the official prompt. That partial artifact and its charge are preserved. Probe-missing handling was added before inspecting any discovery quality outcomes; the checksum-bound resume replayed the last valid scoring boundary and completed all400 questions. Details are in PROTOCOL.md and the retained recovery receipts.
