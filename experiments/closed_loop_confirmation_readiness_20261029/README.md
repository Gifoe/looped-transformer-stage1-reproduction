# Conditional independent-confirmation readiness

This directory audits task definitions while the frozen 3,200/800 calibration study runs. It contains no new benchmark examples, gold labels, model outputs, fitted heads or selected final controller. This preparation does not override the development admission gate.

The installed lm-evaluation-harness definitions were read directly from the audited server environment. Their exact bytes and SHA-256 hashes are stored under `sources/`. The attached excerpts from `lm_eval/api/task.py` establish the default single-space continuation delimiter and character-length normalization, including the exclusion of that delimiter from the normalization denominator.

| Candidate task | Official prompt/continuation convention | Metrics in installed definition |
| --- | --- | --- |
| ARC-Easy | `Question: {question}\nAnswer:`; continue with a space followed by each complete choice text | `acc`, `acc_norm` |
| CommonsenseQA | Question and A–E options in the prompt; continue with a space followed by the option letter | `acc` |

For a possible future confirmation, ARC-Easy would use `acc_norm` as the primary outcome and also report `acc`; CommonsenseQA would use its declared `acc`. Every CommonsenseQA scored continuation is a one-character label before adding the delimiter, so character normalization yields the same prediction. Scoring its full answer text would change the installed task definition.

`scripts/format_observation.py` accepts only the question and choices, without an answer key. `scripts/audit_formatting.py` checks synthetic boundary-space, Unicode and multiline cases against the exact installed Jinja templates. It does not instantiate task objects or download datasets. The audit covers formatting and denominator conventions; actual tokenizer, baseline-logit and selection parity still require validation before running confirmation.

A future confirmation requires passing the frozen development admission rule. If a controller passes, first pin dataset revisions and task/scorer sources, exclude IDs and normalized question texts in the existing exposure index, bind one frozen controller and all baseline configurations, and specify paired tests and complete compute accounting in a final lock. This preparation has not established independence from semantic duplicates or model pretraining. Failed development candidates must not use these tasks for another selection round.
