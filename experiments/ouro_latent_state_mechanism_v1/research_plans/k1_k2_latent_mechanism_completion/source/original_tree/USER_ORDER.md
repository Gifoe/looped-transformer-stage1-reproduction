# Astra → Sol: Complete the K1→K2 Latent Mechanism Investigation

## Research Goal

**Identify and causally explain how Ouro-1.4B transforms its latent representations from Loop 1 to Loop 2, why this transition produces a substantial improvement in reasoning accuracy, and which computational principles can eventually be exploited to improve inference.**

The primary goal is mechanistic understanding, not immediate accuracy optimization.

We must move from observing useful hidden-state changes to identifying the actual recurrent computation that generates and uses them.

## 1. Existing Evidence

Repository: https://github.com/Gifoe/looped-transformer-stage1-reproduction

Starting commit: `1693cd0490282776e2a82c57397c74ad24fbc1ec`

Read all completed E1–E6 reports, implementation code, independent audits, per-question results and counterexamples before designing new experiments.

Existing observations:

- Native Ouro-1.4B improves substantially from K1 to K2 under the fixed ARC-Challenge Joint-Option protocol.
- E2 demonstrates improved answer decodability from K2 hidden states under a frozen decoder.
- E3 identifies causally effective, Q-region-conditioned L10–L16 readout-position attention updates.
- E4 establishes conditional R-state/attention mediation for a subset of natural K2 corrections, with substantial remaining downstream effects.
- E5 reveals depth-dependent, non-monotonic recurrent behavior.
- E6 demonstrates that simple natural latent geometry features have not provided reliable incremental inference utility beyond output-only baselines.

These observations are starting evidence, not a complete K1→K2 explanation.

**Do not repeat completed experiments unless a specific missing causal link or numerical inconsistency requires it.**

## 2. Experiment A — Information Availability vs. Readability

### Scientific Question

Does K1 already contain computationally accessible answer-relevant information that K2 reorganizes for effective readout, or does the second recurrent pass establish additional usable representations through further computation?

### Required Investigation

Trace natural hidden representations at the recurrent boundary and selected physical layers.

Compare K1 and K2 using identical decoders, cross-loop probe transfer, candidate-relative representation analysis and matched causal interventions.

Reuse the existing E2 results before introducing new probes.

Distinguish:

- Information decodable from a frozen representation.
- Information utilized by the native computation.
- Changes in representation geometry or readout compatibility.
- Information that becomes accessible only after further recurrent processing.

Use Train-only probe fitting and frozen evaluation. Control candidate mappings, option-label biases, the previously discovered extra E-axis, probe capacity, representation scale and intervention dose.

**Deliverable:** Determine which explanations are supported or rejected, and identify the earliest observable representation changes associated with natural K1→K2 corrections.

Do not equate failed linear decoding with information absence, or successful decoding with native causal use.

## 3. Experiment B — Causal K1→K2 State-Transition Tracing

### Scientific Question

Which computations actually transform the end-of-K1 representation into the decision-relevant states observed during K2?

### Required Investigation

Start at the true recurrent boundary:

K1 final state → K2 input → K2 intermediate computations → K2 decision.

Trace the actual K2 computation using targeted interventions on:

- Recurrent-boundary states.
- Question/option-region representations.
- Readout-position residual states.
- Attention and MLP updates along the identified upstream path.

Prioritize the previously supported L10–L16 attention-update family, but do not assume it is the unique writer.

Use the existing E3 evidence to construct competing explanations:

**H1:** K2 mainly improves access to information already encoded in K1.

**H2:** K2 performs additional information integration that produces newly usable representations.

**H3:** The gain depends on interactions among several residual, attention and downstream paths rather than one isolated component.

Design interventions that distinguish these explanations.

Use same-question donor states, bidirectional restoration and rollback, identity controls, matched-magnitude perturbations and unrelated-direction controls.

Distinguish natural K1→K2 differences from E3's counterfactual N−P reference differences.

Track both current-question source-region influence and readout-position changes without assuming a token's causal influence identifies the original semantic source of information.

**Deliverable:** Identify a minimal, evidence-supported computational path connecting the K1 recurrent boundary to specific beneficial K2 state transformations.

A distributed path is acceptable if individual writers cannot be isolated. Do not manufacture a single-layer mechanism.

## 4. Experiment C — Natural Correction Mediation

### Scientific Question

Do the identified upstream computations genuinely participate in natural K1→K2 corrections?

### Required Investigation

Connect the upstream mechanism from Experiment B to the previously established L18 residual/attention readout effects.

The immediate priority is a same-question, same-receiver causal mediation experiment:

1. Start from natural K2.
2. Apply a controlled removal of the candidate L10–L16 update.
3. Capture the resulting actual L18 R residual and attention computation.
4. Independently restore the relevant L18 residual or attention mediator while retaining the upstream intervention.
5. Recompute the complete remaining network.
6. Compare restoration, direct bypass, interaction and unsuccessful recovery against matched controls.

Then test the corresponding forward restoration in a suitable earlier or perturbed receiver.

Preserve all four natural outcome groups:

- K1 wrong → K2 correct.
- K1 correct → K2 wrong.
- Both correct.
- Both wrong.

Report candidate scores, actual predictions, ties, rescue/damage, margins, paired uncertainty and individual counterexamples.

Distinguish conditional causal mediation from universal necessity or sufficiency.

**Deliverable:** Establish which natural corrections are sensitive to the identified path, which can be restored through the proposed mediator, and which remain unexplained.

Do not calculate a purported percentage of all reasoning explained from transplant rescue counts.

## 5. Scientific Execution Strategy

Astra is the scientific director; Sol is the experimental executor.

Astra must autonomously review findings, formulate competing mechanisms, issue executable experiment orders, evaluate negative results and revise the investigation.

Sol must implement and execute real GPU interventions, perform numerical verification, retain original tensors and per-question evidence, and produce reproducible code and reports.

Prioritize the most discriminating missing experiment rather than running every possible combination.

Proceed through:

**Hypothesis → Causal Experiment → Independent Audit → Scientific Review → Revised Hypothesis.**

Begin with existing data and targeted Train discovery. Freeze specific interventions and evaluation criteria before confirmation.

Use the complete appropriate cohort when feasible, with Train non-demonstrations for development and separately reported Validation confirmation. Historical Test exposure must remain explicit; Test cannot be repeatedly used for mechanism selection.

Maintain the original model identity, BF16/SDPA and Joint-Option scoring protocol. Re-establish same-hardware native baselines when needed; never mix historical A6000, RTX4090 and RTX5090 predictions as numerically identical.

If extending to Full-Answer scoring, establish a separate matched native baseline and report the results independently.

No model-weight training, label-dependent inference, destructive changes to previous results or automatic external benchmark expansion.

Do not launch E6-C or another accuracy-optimization search until the mechanism investigation produces a concrete justification.

## 6. Research Deliverables

Create a new research directory:

`experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_latent_mechanism_completion/`

Include:

- `RESEARCH_HYPOTHESES.md`
- `INFORMATION_AVAILABILITY.md`
- `K1_K2_STATE_TRANSITION.md`
- `UPSTREAM_CAUSAL_TRACING.md`
- `NATURAL_MEDIATION_RESULTS.md`
- `CAUSAL_EVIDENCE_MAP.md`
- `COUNTEREXAMPLES.md`
- `NUMERICAL_AND_SOURCE_AUDIT.md`
- `FINAL_MECHANISTIC_MODEL.md`
- `FINAL_RESEARCH_REPORT.md`

Retain complete scripts, frozen orders, source/checkpoint hashes, data coverage, execution receipts, independent numerical audits and reproducible per-question results.

The final report must answer:

**What is represented after K1? What actually changes during K2? Which recurrent computations cause those changes? How do they affect answer selection? Which natural corrections can be causally attributed to the identified path, and which cannot?**

Finally, identify the specific mechanistic principle that could guide an inference-time method, while distinguishing an experimentally validated control signal from a speculative application.

## 7. Completion Criteria

Do not declare the K1→K2 mechanism completely understood merely because a local intervention improves accuracy or a probe decodes more information.

The research reaches a defensible mechanistic conclusion when it establishes a reproducible causal connection between:

**K1 State → K2 Computation → Decision-Relevant Latent Transformation → Downstream Readout → Natural Correction.**

If competing explanations remain observationally equivalent, explicitly identify the unresolved causal edge and perform the smallest feasible experiment that distinguishes them.

If the evidence cannot resolve an edge, report the exact limit rather than presenting a correlation as a complete explanation.

**Astra: begin by reviewing E1–E6, identifying the single most important missing causal link, and directing Sol to execute the first discriminating experiment. Continue autonomous, evidence-driven iterations until a coherent causal mechanism is established or the remaining uncertainty is explicitly characterized and independently audited.**