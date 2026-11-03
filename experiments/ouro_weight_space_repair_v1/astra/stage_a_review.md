# Astra Stage A mechanism review

Date: 2026-11-03. Status: design review before access to any new experimental results. No GPU jobs launched; no prior ARC test labels, predictions or trajectories inspected. Actual hardware is user-confirmed GB10 at 10.20.11.80, superseding the attachment's RTX 5090. Checkpoint revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. Additional budget: 8 GPU-hours.

## Decision

Proceed with a small causal audit, not an assumed harmful-direction removal. H1, H2 and H3 are not mutually exclusive hypotheses, and the requested measurements alone do not identify them. A correct-to-incorrect transition establishes benchmark harm from the complete recurrent transition, but does not identify a shared harmful weight component. Hidden cosine similarity, PCA concentration and large projection norms are descriptive evidence only. A successful endpoint interpolation is compatible with both step-scale sensitivity and readout sensitivity.

The strongest attainable conclusion within this budget is operational: a fixed component intervention, specified without fitting labels, improves held-out questions beyond tested scalar and readout controls, and its exact shared-weight realization preserves that benefit across loop depths. Anything weaker must be reported with its limitation. Failure of three damping coefficients does not prove that no scalar repair exists. Failure to find a direction does not prove that no harmful component exists.

## Primary-source context and novelty

The MIT blog reports sparse MLP output-column edits addressing Qwen's generated-text repetition, and a negative result for a smaller model whose failures were ordinary reasoning errors. This is motivation for conditional repair, not evidence that Ouro has the same defect. Repeated autoregressive text and repeated latent application of shared weights are different mechanisms. Do not inherit MIT's layer indices or column counts. [Primary blog](https://rikiyat.github.io/blog/weight-space-programming/)

Experiment 8's README specifies training examples for feedback, prohibition of backward/optimizers/iterative fitting, and BF16-cast evaluation. It also distinguishes touched coordinates from coordinates that actually change after BF16 rounding. Its successful sparse edit is not evidence that arbitrary low-rank edits work. I inspected the README only, not per-problem benchmark files or run trajectories. [Experiment 8 protocol](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/README.md)

Ouro explicitly reuses a shared stack across recurrent steps. Consequently, a fifth-loop-only lesion is an intervention on an execution instance, whereas editing one physical matrix changes every invocation. Transfer from the former to the latter is an empirical requirement. [Ouro project](https://ouro-llm.github.io/)

The Right Direction, Wrong Step abstract already studies reference utility along the native displacement, showing that an initially beneficial direction can have a harmful full step. Thus, interpolation rescue is not novel here. The new contribution would need to establish a transferable, component-specific causal repair beyond that account. The abstract's findings concern selected failures and reference utility; they cannot be treated as an expected unconditional ARC accuracy improvement. Full-text per-question examples were deliberately not opened. [Primary preprint](https://arxiv.org/abs/2609.16665)

## Source-code issues that change the experiment

Inspected only `scripts/common.py`, `scripts/prepare_data.py`, `scripts/score_arc.py` under `experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026`, and `experiments/looped_stage1_trajectory_audit_seed2021_v1/sources/modeling_ouro.py`.

1. **Boundary normalization participates in recurrence.** `modeling_ouro.py:592–607` assigns `hidden_states = self.norm(hidden_states)` and then reuses that normalized state in the next loop. The returned loop states are already normalized. Define exactly which coordinate system is interpolated; never silently apply the final norm twice. Cache both the raw pre-boundary state and the actual normalized feedback state when feasible.
2. **There is branch normalization after output projection.** Attention `o_proj` is followed by `input_layernorm_2`; MLP `down_proj` is followed by `post_attention_layernorm_2` before residual addition (`:401–422`). A hook that subtracts a direction after these normalizations is not the activation realization of a weight edit. Apply the exact projection-output intervention before its following norm. Uniform scaling of a branch's output can be nearly canceled by RMSNorm; a scalar output-matrix shrink is not a useful mechanism-equivalent control.
3. **The score is complete-candidate likelihood normalized by character length.** `score_arc.py` scores all continuation positions, and `candidate_result` divides by `choice_char_lengths`, not token count. Diagnostics at the last prompt token alone do not identify the benchmark margin. Reuse all original candidate strings, tokenization, positions and denominator rules. Define margin as normalized gold score minus the maximum other normalized score. Softmax over normalized candidate scores is a diagnostic weight, not a calibrated answer probability.
4. **Demonstrations may vary per query.** `requests_for` calls `fewshot_context` repeatedly after one seeded sampler initialization. Establish the actual sampler behavior from code before claiming a fixed 25-demo set. A varying 25-shot sampler can eventually use almost the whole train split. Discovery exclusion must cover the full union of demos used in the new experiment. Do not open prior test prompts to resolve this; derive the policy and demo-ID manifest from training data and code. If preserving the old policy conflicts with a usable disjoint discovery pool, record that conflict and explicitly resolve it before running. A frozen disjoint demonstration pool is a methodological change and must not be described as exact old-prompt parity. Whichever construction is used, compare every new method on identical prompts.
5. **Do not run old entry points.** Old preparation/evaluation `main()` functions read old test data and budget files. Reuse pure logic, not execution paths that access prohibited files. Create an independent additional-budget ledger.

## Minimal informative protocol

### A0: freeze and validate

Freeze demo IDs, train discovery/selection IDs, seed 2021, scoring, precision, coefficient grids and candidate cap before outcomes. Suggested initial scale is 64 discovery plus 64 selection questions, with an optional predeclared expansion to 128 each if too few discordant cases; sizes are contingent on measured GB10 throughput. Official validation is used only after candidate definitions freeze. Keep test physically inaccessible to diagnostic code. Split by question, never by answer candidate/token.

Run a tiny non-test parity check comparing an ordinary K=4 native forward with the new no-op path, including all candidate scores and predictions. Prove that K=3/4/5 states obtained during one K=6 pass agree with separate K runs on the same examples; only then reuse intermediate outputs. Check the early-exit gate does not alter the requested fixed-depth score. No gradients, optimizer, train mode or learned probe.

### A1: native trajectories and inexpensive readout paths

Obtain K=3,4,5,6 baseline scores for the complete initial discovery and selection sets. Record all four transition categories and margins, not only selected regressions. Retain answer-position states and aggregated branch statistics; do not dump all activations for all 25-shot tokens.

For the primary 4-to-5 transition let s4 and s5 be actual normalized loop boundary states and d=s5-s4. A readout-only path is H(s4+a*d), with H the unchanged LM head and original candidate scoring; do not re-normalize these returned states. Use a fixed grid a in {0,0.25,0.5,1}; add a predeclared small positive step such as 0.125 only if precision checks resolve it. a=0 is an early-state control, never evidence of useful fifth-loop computation. Report whether an interior point beats BOTH endpoints in margin and whether its accuracy changes agree.

This normalized chord changes both direction and norm. Separately, on a small subset, interpolate raw pre-boundary states and apply the unchanged boundary norm exactly once; label that path separately. Differences between these paths reveal normalization sensitivity. Do not merge them into one “step size” result.

Endpoint-only interpolation can be computed without rerunning the transformer. It tests accessible readout geometry, not improved dynamics. Local finite-difference improvement plus a negative full-step gain is finite-step-failure evidence along the specified path; an accuracy flip alone cannot establish local direction quality.

### A2: carry the intervention into recurrence

On the identical examples, feed the chosen fixed interpolated loop-5 boundary into loop 6 and score K=6. Apply the intervention to all sequence positions when claiming a dynamical intervention. Compare with (i) native K=6, (ii) reading the interpolated boundary without recurrence, and (iii) unchanged recurrence with a readout-only mixture at K=6. Endpoint-only rescue that disappears under recurrence supports readout-accessibility sensitivity; it does not prove all useful information remains in native s5.

Existing fixed/adaptive LoopCD at K=4 should be computed from the same native trajectory and original implementation. It is a useful readout baseline but not a test that the LM head itself is defective: it accesses additional states. Under the no-probe/no-fitting constraints, cleanly separating all possible H3 explanations from H1/H2 is impossible. Preserve an “H1/H3 unresolved” category.

### A3: localized causal screening, then a direction-specific test

Choose a few representative 4-to-5 regressions AND stable-correct, stable-wrong, and 4-to-5 improvement cases using a preregistered deterministic rule. Layerwise output norms/alignment may prioritize at most two physical layers for expensive interventions; they are not causal rankings. Start with coarse group lesions if the descriptive signal is diffuse. Count each layer/group examined in the search ledger.

Screen attention and MLP separately on loop 5. For an output projection W and its input a, replace y=W*a by y-alpha*P*y BEFORE the branch output norm, then allow all subsequent computation to run. P must be defined without supervised fitting: use a small, frozen set of weight singular directions or label-blind PCA directions of discovery outputs, rank 1 initially. Labels may assess a finite candidate set, not optimize a vector by minimizing benchmark loss. Do not learn a gold-minus-distractor direction or fit a probe under the “no supervised parameter fitting” constraint.

Use alpha in {0.5,1}, with alpha=0 as the exact no-op. Compare the directional intervention against: a scalar total-loop chord control; a scalar post-branch-norm lesion with matched residual perturbation magnitude; an orthogonal/random direction with matched perturbation magnitude; and a norm-preserving directional intervention. For the latter, on cached same-state outputs, rescale the surgically modified residual update to the original update norm. This is a diagnostic control, not a proposed weight edit. If it retains improvement, simple shrinkage becomes a less sufficient account. Declare precise norm locations, because pre-projection and post-norm matching differ.

The minimum credible H2 signal is consistent benefit from a specific non-collinear removal, beyond these bounded scalar controls, on nonselected questions without excessive regressions. Reversing the edit sign or adding back the removed component should reverse/erase the rescue on the same computation when numerically resolvable. Random directions should not systematically do as well. One random draw is only a sanity check; three seeded draws give a limited control, not a null distribution strong enough for a broad significance claim.

### A4: require shared-weight transfer

The same fixed projection-output intervention must next operate on every invocation of that physical module, across all loops and all positions. Only if this transfers to held-out train selection should it become a permanent weight candidate. A fifth-loop-only improvement accompanied by all-loop degradation is a failed shared-weight repair, even if its causal localization is informative.

Freeze at most one or two serious edit definitions, their strengths, controls and selection rule before official validation. Official validation may select at most one primary edit, or select none. No new directions, layers, strengths or changed story after validation feedback. Primary test comparison remains edited K=5 versus original K=5, with K=3/4/6 secondary. If no edit passes the gates, stop Stage B/C and report a negative result; do not fabricate an edited model to populate deliverables.

## Conditional weight families

For a unit output direction u, the projection-output intervention y'=(I-alpha*u*u^T)y corresponds in exact arithmetic to W'=W-alpha*u*(u^T W). For orthonormal U with r columns, W'=W-alpha*U*(U^T W) has update rank at most r. This is meaningful at the pre-output-norm location above. Rank 4 is justified only if rank 1 leaves a reproducible multi-directional component; it is not an automatic extra search arm.

A sparse variant selects a fixed set S of input columns and uses W'=W-alpha*u*(u^T W)*D_S. The corresponding activation control removes alpha*u*sum_{j in S}[(u^T W_j)*a_j]. Column scores may use unsigned label-blind activation moments times directional loading; do not select columns by supervised loss fitting. Require this precise sparse activation intervention to work before writing its weights. Begin in one layer; add a second only if independent causal evidence shows the first is insufficient. No embeddings, head, final norm or architecture edits.

“Low rank” does not mean “few changed weights”: a dense rank-one update usually touches an entire matrix. Report the number of stored factors, intended support coordinates, BF16 coordinates actually changed, total original unique parameters, fraction changed, Frobenius norm and relative norm per tensor. Check tied tensors are counted once.

## BF16 and statistical controls

- Compute statistics, SVD and proposed deltas in FP32 (or FP64 summaries); materialize the candidate through the exact BF16 rounding path used in evaluation. Count zeroed-by-rounding entries. Record base and edited hashes. FP32 factors alone do not reproduce the evaluated tensor bit for bit unless the cast/add order is fixed.
- Compare the actual rounded weight edit with its activation emulation on a small parity set. Differences caused by BF16 GEMM/normalization order can occur even when algebra is exact. Require the effect to exceed no-op/repeat numerical variability; report unresolved discrepancies rather than expanding numerical tolerances until they pass.
- Random edits must match tensor support, rank where feasible, and ACTUAL BF16 delta norm; also report their functional activation perturbation norm. Equal nominal alpha is not norm matching. Keep deterministic seeds separate from seed 2021 data construction.
- Use paired question-level bootstrap intervals for accuracy differences and margins, and exact McNemar/binomial tests on correction versus regression counts. Do not bootstrap candidate tokens as independent observations. Report raw discordant counts. Selection-conditioned discoveries are exploratory; their p-values are not confirmatory evidence.
- Freeze one primary test contrast. Secondary depths and controls require multiplicity qualification, not a claim selected from the best test K. One ARC checkpoint and one benchmark can establish a local mechanism and replication target, not general training-free self-improvement.

## Budget and stop rule

Time a few real 25-shot examples on GB10 before choosing sample sizes. Reserve the final complete K<=6 edited/native trajectories, controls actually required on test, and validation before spending the remainder on diagnosis. If that reserve alone exceeds the budget, reduce diagnostic breadth or stop before opening test; never silently truncate final 1,172-question evaluation. A six-loop pass gives intermediate depths only after parity is demonstrated, and diagnostic cached logits can avoid redundant transformer runs.

Stop weight editing if there are too few reproducible harmful transitions, scalar/readout controls account for the observed gain, direction-specific effects fail held-out train replication, or global shared edits destroy the benefit. In these cases the correct output is a documented failure to establish H2/shared repair, not the stronger claim that H2 is absent.

The proposed direction is not yet demonstrably publishable. Its default outcome risks being a narrow application of known interpolation and projection methods. The publishable discriminator would be a shared, causally validated component whose sparse fixed weight repair transfers to unseen questions and additional loops beyond step-scale/readout controls, with a clear account of the branch-normalization and shared-invocation effects. No existing result in this review establishes that discriminator.
