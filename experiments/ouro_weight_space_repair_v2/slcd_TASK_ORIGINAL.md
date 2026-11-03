# Autonomous Research: Self-Improving Training-Free LoopCD

## 0. Core Mission

You are an autonomous AI research agent.

Your task is to discover, implement, evaluate, and iteratively improve training-free inference methods that outperform the original LoopCD on Ouro-1.4B.

**Your primary objective is accuracy improvement, not methodological novelty.**

You have substantial autonomy to inspect the repository, design alternative algorithms, modify experimental code, execute experiments, analyze failures, and determine subsequent research steps.

Do not merely implement a fixed list of methods. Use experimental evidence to decide what to investigate next.

The research process should follow:

**Hypothesis → Implementation → Experiment → Failure Analysis → Revised Hypothesis → Improved Method → Validation**

Repeat this loop using development data until a promising method is obtained or the computational budget is exhausted.

### Absolute Benchmark Priority

**ARC-Challenge is the mandatory first benchmark.**

You MUST complete development, freeze the selected method, and evaluate the complete ARC-Challenge test set before running any other benchmark.

The current target is:

**Beat 717/1172 = 61.177% on ARC-Challenge.**

Only after achieving this target may you evaluate HellaSwag, OpenBookQA, or other benchmarks.

If the frozen method fails to exceed 717 correct answers, stop further benchmark evaluation and report the result.

Do not use ARC test labels or results for iterative parameter tuning.

---

## 1. Repository and Environment

Repository:

https://github.com/Gifoe/looped-transformer-stage1-reproduction

Relevant previous experiments:

- `experiments/looped_stage1_trajectory_audit_seed2021_v1/`
- `experiments/ouro_weight_space_repair_v1/`
- `experiments/ouro_weight_space_repair_v2/`

Create:

`experiments/ouro_loopcd_self_improvement_v1/`

Model:

- ByteDance/Ouro-1.4B
- Checkpoint revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`
- BF16
- NVIDIA GB10
- Existing fixed 25-shot ARC-Challenge protocol
- lm-evaluation-harness `acc_norm`
- Seed 2021

Do not train or modify model weights.

No gradient descent, learned adapters, fine-tuning, or additional trained neural networks.

Reuse the existing scoring implementation and environment wherever possible.

Preserve all previous experiments.

---

## 2. Existing Experimental Evidence

Our previous full ARC-Challenge evaluation produced:

| Method | Correct / 1172 | Accuracy |
|---|---:|---:|
| Ouro K3 | 696 | 59.386% |
| Ouro K4 | 702 | 59.898% |
| Ouro K5 | 680 | 58.020% |
| Ouro K6 | 680 | 58.020% |
| Fixed LoopCD K4 | 717 | 61.177% |
| Adaptive LoopCD K4 | 716 | 61.092% |

Fixed LoopCD:

- 41 corrections
- 26 regressions
- Net +15

Adaptive LoopCD:

- 52 corrections
- 38 regressions
- Net +14

These results suggest that LoopCD generates useful corrections but also introduces unnecessary harmful changes.

Further analysis of saved per-question candidate scores suggested:

\[
S_{\text{new}}(c)
=
S_{\text{Fixed}}(c)
+
\gamma[S_4(c)-S_5(c)]
\]

With gamma=1, retrospective ARC test rescoring produced 726/1172, compared with 717/1172 for Fixed LoopCD.

This result has already been inspected and must be considered exploratory.

Do not use this result to tune gamma or choose the final method.

The purpose of the new experiment is to investigate whether this or another modification can yield transferable, reproducible accuracy improvements.

---

## 3. Stage A — Reconstruct Existing Results

Before developing new methods:

1. Inspect the current LoopCD implementations.
2. Read previous ARC-Challenge experimental reports.
3. Reconstruct saved native K1–K6 candidate scores where available.
4. Verify Fixed and Adaptive LoopCD score parity.
5. Inspect the previous correction/regression statistics.
6. Identify which information is available without additional GPU inference.

Use development splits for method analysis.

Do not inspect or reanalyze ARC test labels, per-question test outcomes, or candidate-level test scores during method development.

The historical aggregate results above are background evidence only.

Ensure all new methods use the same complete-candidate scoring protocol.

---

## 4. Stage B — Initial Research Directions

Begin with the following three method families.

These are starting points, not mandatory final solutions.

You may abandon unpromising methods and develop alternatives based on development evidence.

### A. Two-Sided LoopCD

Original LoopCD:

\[
z_{\mathrm{CD}}
=
z_4+\omega(z_4-z_1)
\]

Potential extension:

\[
z_{\mathrm{new}}
=
z_{\mathrm{CD}}-\gamma(z_5-z_4)
\]

Investigate whether late-loop changes contain information that can reduce harmful extrapolation.

Implement both:

**A1. Score-level correction**

\[
S_{\mathrm{new}}(c)
=
S_{\mathrm{CD}}(c)
+
\gamma[S_4(c)-S_5(c)]
\]

**A2. Token-logit-level correction**

\[
z_{\mathrm{new},t}
=
z_{\mathrm{CD},t}
+
\gamma(z_{4,t}-z_{5,t})
\]

For A2, apply the modified logits before log-softmax and compute the complete candidate likelihood correctly.

Score-level and logit-level methods are not mathematically equivalent.

Initial gamma candidates:

`[0, 0.25, 0.5, 0.75, 1.0]`

Gamma=0 must exactly reproduce the original Fixed LoopCD.

You may refine the search based on training/development results, but keep the number of candidates bounded.

### B. Orthogonal Guidance

Investigate whether the full contrastive direction is necessary.

Let:

\[
d=z_4-z_1
\]

Decompose this direction into parallel and orthogonal components relative to the current logits.

Evaluate whether an orthogonal-only intervention can improve candidate ranking while reducing harmful corrections.

Implement the original paper's decomposition accurately and compare with Fixed and Adaptive LoopCD.

This family is attractive because it does not require an additional recurrent iteration.

### C. Candidate-Aware Adaptive Guidance

Investigate whether the guidance strength should depend on complete answer candidate information.

Potential inference-time signals:

- Candidate score margin
- Rank disagreement between loops
- Contrastive score displacement
- Candidate score trajectory
- Cross-loop confidence variation
- Magnitude and direction of late-loop updates

Design deterministic, training-free adjustment rules.

The method must not use gold labels during inference.

Avoid fitting a classifier or training an auxiliary gating network.

The objective is to reduce harmful corrections without sacrificing too many beneficial corrections.

---

## 5. Stage C — Autonomous Self-Iteration

This is the central part of the project.

You are NOT required to follow a predetermined experimental sequence after the initial baselines.

Instead, autonomously develop methods using evidence from the previous experiment.

### Iteration Procedure

For each research iteration:

**Step 1 — Analyze the current failure.**

Determine:

- Which examples improved?
- Which examples degraded?
- Whether the change improves mean candidate margins
- Whether corrections generalize from training discovery to selection/validation
- Whether improvements depend on a narrow coefficient interval
- Whether the method requires additional recurrent computation

**Step 2 — Form a specific hypothesis.**

Examples:

- Late-loop corrections improve performance only when K4/K5 disagree.
- Guidance strength should depend on candidate-level uncertainty.
- Orthogonal guidance avoids some harmful ranking changes.
- Early-loop and late-loop score differences contain complementary information.

Do not assume any of these hypotheses is correct.

**Step 3 — Implement a minimal intervention.**

Prefer simple, interpretable transformations of available logits or candidate scores.

Avoid unnecessary model complexity.

**Step 4 — Evaluate on development data.**

Compare against the original Fixed LoopCD on identical questions and prompts.

Report:

- Correct count
- Accuracy
- Corrections
- Regressions
- Net corrections
- Margin changes
- Additional computation

**Step 5 — Decide what to do next.**

If results improve consistently across development subsets, retain the method.

If improvements are unstable, analyze the underlying cause and revise the hypothesis.

If a method fails repeatedly, abandon that branch instead of continuing blind parameter search.

### Search Freedom

You may:

- Develop new guidance equations.
- Combine promising components.
- Change guidance strength.
- Use multiple recurrent states.
- Explore score-space and logit-space transformations.
- Design deterministic inference-time decision rules.
- Perform targeted ablations.
- Replace unsuccessful ideas with more promising ones.

You must document every substantive method revision.

### Search Constraints

- No gradient-based training.
- No use of ARC test outcomes for method selection.
- No unlimited hyperparameter sweeps.
- No massive combinations of arbitrary heuristics.
- No post-hoc test-set winner selection.
- No modification of the original model weights.

Use the training discovery and independent selection subsets for iterative experimentation.

Reserve official ARC validation for final method comparison and selection.

Do not repeatedly adapt the method to official validation feedback.

You may conduct up to six substantial development iterations within the compute budget, followed by one frozen validation-selection stage.

The final algorithm should be motivated by experimental evidence, not by the number of experiments attempted.

---

## 6. Stage D — Select the Best Method

After self-iteration, compare the most promising methods on official ARC validation.

Selection priorities:

1. Accuracy improvement over Fixed LoopCD
2. Net correction count
3. Consistency across development subsets
4. Paired uncertainty
5. Computational overhead
6. Algorithm simplicity

Select ONE primary method and optionally ONE secondary method.

Freeze the method definitions and all numerical coefficients before final test evaluation.

Create:

`FINAL_METHOD_LOCK.json`

The lock must include:

- Model checkpoint
- Selected algorithm and source hash
- Selected coefficients
- Loop depths
- Evaluation prompts
- Dataset revision
- Scoring protocol
- Fixed random seed
- Method selection evidence
- Final test access authorization

After this lock is created, do not change the algorithm based on test feedback.

---

## 7. Stage E — Full ARC-Challenge Test

This stage is mandatory and must happen BEFORE any other benchmark.

Evaluate all 1,172 official ARC-Challenge test questions.

Use the same fixed 25-shot protocol as V1.

Evaluate:

1. Native Ouro K4
2. Original Fixed LoopCD
3. Original Adaptive LoopCD
4. Selected improved LoopCD
5. Optional frozen secondary method

Reuse existing baseline outputs only after verifying exact prompt, candidate, checkpoint, scoring, and environment compatibility.

Primary comparison:

**Selected improved LoopCD versus original Fixed LoopCD.**

The minimum performance gate is:

\[
\mathrm{Correct}_{new} > 717
\]

Compute:

- Accuracy
- Correct count
- Corrections/regressions
- Paired bootstrap 95% CI
- Exact McNemar p-value
- Additional recurrent forward cost
- Paired prediction changes

Distinguish a positive point estimate from statistically supported improvement.

The ARC test has been used in previous research and the exploratory two-sided result has already been inspected. Report it as a previously used regression benchmark rather than an unseen blind test.

### Strict Post-Test Decision

**Case A: Improved method scores above 717/1172**

Report the result and its uncertainty.

Only then proceed to Stage F.

**Case B: Improved method scores 717/1172 or below**

Stop immediately.

Do NOT run HellaSwag, OpenBookQA, or any other benchmark.

Explain why the proposed methods failed to outperform LoopCD.

Do NOT adjust coefficients or choose another candidate using ARC test results and rerun the test.

Do not turn the test into another development set.

---

## 8. Stage F — Cross-Benchmark Evaluation (Conditional)

This stage is authorized ONLY if Stage E exceeds 717/1172.

Evaluate the frozen selected method on:

1. HellaSwag
2. OpenBookQA, if the remaining compute budget allows

Use the official task definitions and consistent baseline prompting.

Run:

- Native Ouro
- Original Fixed LoopCD
- Original Adaptive LoopCD
- Selected improved LoopCD

Do not change the selected method based on these results.

Report whether the ARC improvement generalizes to other tasks.

If the method fails to generalize, report the benchmark dependence without introducing new test-driven modifications.

---

## 9. Computational Budget

Available hardware: NVIDIA GB10.

Initial additional GPU-worker budget: 6 hours.

Prioritize:

1. Existing-data CPU experimentation
2. ARC development and method refinement
3. Frozen ARC validation
4. Complete ARC test
5. Conditional external benchmark evaluation

Reserve sufficient compute for the full ARC test before spending additional resources on method search.

You may optimize forward reuse, cache storage, vectorization, and experiment scheduling while preserving numerical equivalence.

Record all actual GPU-worker consumption, including failed attempts.

Do not claim runtime improvements based on shared-forward trajectory evaluations.

If the budget is exhausted before a valid complete ARC test can be run, report incomplete execution rather than fabricating success or using partial test accuracy.

---

## 10. Required Deliverables

Create:

`experiments/ouro_loopcd_self_improvement_v1/`

Required outputs:

- `REPORT_ZH.md`
- `README.md`
- `SELF_ITERATION_LOG.md`
- `hypotheses_and_decisions.csv`
- `development_results.csv`
- `ablation_results.csv`
- `FINAL_METHOD_LOCK.json`
- `arc_full_test_results.csv`
- `paired_transitions.csv`
- `compute_costs.csv`
- `accuracy_comparison.pdf`
- `rescue_regression_analysis.pdf`
- Complete implementation and reproduction scripts

### SELF_ITERATION_LOG.md

For every research iteration, record:

1. Current hypothesis
2. Evidence motivating the hypothesis
3. Algorithmic modification
4. Experimental configuration
5. Observed results
6. Why the method improved or failed
7. Decision to retain, revise, or abandon
8. Next research action

The purpose is to preserve an auditable scientific reasoning and experimentation process.

### Final Chinese Report

Answer:

**Q1.** Why does original LoopCD produce harmful corrections?

**Q2.** Which improvement approaches were implemented?

**Q3.** What did autonomous iteration discover?

**Q4.** Which method performed best on development data?

**Q5.** Did the frozen method exceed 717/1172 on ARC-Challenge?

**Q6.** How many errors were corrected, and how many new errors were introduced?

**Q7.** Is the gain statistically reliable and computationally worthwhile?

**Q8.** If ARC improved, did the method generalize beyond ARC?

**Q9.** Should we continue this research direction?

---

## 11. Final Execution Instruction

Act as an autonomous experimental scientist.

You are authorized to inspect existing experiments, implement alternative algorithms, execute GPU experiments, diagnose failures, and autonomously refine the research direction using development evidence.

Do not stop after merely reproducing existing baselines.

Do not implement only the first proposed formula and declare the task complete.

Use the available development budget to search intelligently for a better training-free inference method.

However:

**The full ARC-Challenge test is the mandatory first final benchmark and the strict gateway to every other benchmark.**

Freeze the method before that test.

If accuracy does not exceed 717/1172, stop and report the negative result.

If it exceeds 717/1172, proceed to cross-benchmark validation without further tuning.

Commit all reproducible artifacts to a new GitHub branch and provide the final repository path, exact numerical results, and research conclusion.