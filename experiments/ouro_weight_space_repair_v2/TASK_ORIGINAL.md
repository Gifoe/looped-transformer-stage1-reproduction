# Experiment V2: MIT-Style Weight Editing for Ouro

## Goal

Implement and evaluate two training-free models:

1. **Edited Ouro** — Ouro-1.4B with directly modified weights.
2. **Edited Ouro + LoopCD** — the same edited Ouro checkpoint with LoopCD applied at inference.

The primary objective is to improve ARC-Challenge accuracy.

Do not spend this experiment proving an entirely new mechanism. Adapt MIT's Weight-Space Programming approach and determine empirically whether it improves Ouro.

## Repository and Environment

Repository:
https://github.com/Gifoe/looped-transformer-stage1-reproduction

Previous experiment:
`experiments/ouro_weight_space_repair_v1/`

MIT reference:
https://github.com/RikiyaT/lm_understand-logs

Create:
`experiments/ouro_weight_space_repair_v2/`

Reuse:

- ByteDance/Ouro-1.4B, checkpoint revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`
- Existing NVIDIA GB10 server environment
- BF16 model inference
- ARC-Challenge, 25-shot, `acc_norm`
- The V1 fixed-demonstration protocol
- Existing model scoring and intervention utilities

Preserve the previous experiment without modification.

## Step 1 — Discover Candidate Edits

Use only the ARC training split for edit discovery and candidate optimization.

Reuse the V1 data separation where possible. You may expand the training discovery pool using previously unused training examples, while preserving separate selection data.

Collect K3–K6 recurrent states and identify samples with:

- Correct-to-wrong transitions
- Wrong-to-correct transitions
- Strongly decreasing correct-answer margins
- Strongly increasing correct-answer margins

Use these samples to identify candidate harmful MLP/Attention output components.

Do not restrict analysis to the three correct-to-wrong examples found in V1.

Prioritize weight modifications that are computationally cheap and structurally interpretable.

## Step 2 — Actually Perform Weight Editing

You MUST implement and evaluate at least one candidate from each family:

**A. Sparse MLP Editing**

Modify selected columns of `down_proj` in one or two physical layers.

Explore a small, justified selection of column counts, such as Top-32, Top-64, or Top-128.

**B. Low-Rank Direction Editing**

Construct rank-1 or rank-4 interventions:

\[
W'=W-\alpha U(U^\top W)
\]

Use forward-derived activation statistics to identify directions. The direction-selection method and data must be documented.

You may inspect more layers and candidate directions when justified by training results.

Important constraints:

- No backward passes or gradient descent.
- No optimizer-based fine-tuning.
- No trained adapter or additional neural network.
- Training-set feedback may be used to select edits.
- Do not modify embeddings, LM head, or tokenizer.
- Edit real model weights, not only temporary activation hooks.
- Apply identical edited physical weights across every recurrent invocation.
- Verify that BF16 weights actually changed.
- Preserve the original checkpoint.

Ouro uses branch normalization after output projections. Account for this when constructing and verifying edits. Uniform scaling of an output matrix is not equivalent to post-normalization residual damping.

**Do not stop merely because the V1 broad-branch damping experiment failed.** This experiment is specifically intended to test actual sparse and directional weight edits.

If every tested candidate degrades accuracy, report the negative result after executing them.

## Step 3 — Select One Edited Checkpoint

Evaluate candidate edits on training discovery and selection subsets.

Use the official validation set only for final candidate selection.

Prioritize K4 accuracy, while recording K3, K5, and K6 results.

For a limited number of candidates, consider:

- Single-layer edits
- Two-layer edits
- Sparse-column edits
- Rank-1/rank-4 edits
- Different strengths

Avoid unconstrained search.

Select exactly ONE edited checkpoint, even if its validation performance is inferior to the original model. Preserve the strongest attempted candidate as an experimental result, not as a claimed improvement.

No validation/test-driven changes after checkpoint selection.

## Step 4 — Implement the Two Final Models

### Model 1: Edited Ouro

Use the selected edited weights with ordinary Ouro inference.

Evaluate K3, K4, K5, K6.

Primary metric: ARC-Challenge `acc_norm`.

### Model 2: Edited Ouro + LoopCD

Load exactly the same edited weights and apply the existing LoopCD implementation.

Evaluate:

- Edited Ouro + Fixed LoopCD, omega=0.5, K4
- Edited Ouro + Adaptive LoopCD, cap=1.0, K4

The Fixed combination is the predeclared primary combined model. Adaptive is a secondary comparison.

Do not tune LoopCD coefficients using the test set.

Check that the combination is computed from the **edited model's own hidden states and logits**, not the original model's cached outputs.

## Step 5 — Final Evaluation

After freezing the edited weights and methods, run the complete 1,172-question official ARC-Challenge test set.

Report:

| Method | K3 | K4 | K5 | K6 |
|---|---|---|---|---|
| Original Ouro | | | | |
| Edited Ouro | | | | |
| Original Ouro + LoopCD | | | | |
| Edited Ouro + LoopCD | | | | |

For LoopCD, report only supported depths. Do not fill unsupported cells with inferred results.

Compare edited and original methods on identical prompts.

Report:

- Accuracy and correct-question counts
- Corrections and regressions
- Net improvement in percentage points
- Paired bootstrap confidence intervals
- McNemar tests
- Changed parameter counts and editing norms
- Whether improvements generalize beyond edit-selection examples

This official test set has been used in previous work. Keep it as a locked regression benchmark, not a newly blind benchmark.

## Compute Budget

Use the existing GB10 server.

Initial additional budget: 6 GPU-worker hours.

Use inexpensive screening before full candidate evaluation. Allocate most computation to actually testing weight edits.

Do not spend the entire budget repeating V1's broad scalar-lesion experiments.

## Required Deliverables

Create:

- `REPORT_ZH.md`
- `edit_search_results.csv`
- `selected_edit_manifest.json`
- `edited_weights/` containing the selected weight delta or a reproducible patch
- `final_comparison.csv`
- `paired_transitions.csv`
- `accuracy_by_loop.pdf`
- Reproduction scripts and complete logs

The Chinese report must answer:

1. Which weights were edited, and why?
2. How many BF16 parameters actually changed?
3. Which candidate edit performed best on validation?
4. Does Edited Ouro improve accuracy?
5. Does Edited Ouro + LoopCD outperform the original LoopCD?
6. Are improvements consistent across recurrent depths?
7. If editing fails, which candidates were actually tested and how much did they degrade?

## Execution Instruction

Execute the complete experiment on the server.

You have autonomy to inspect the Ouro architecture, select reasonable candidate edits, modify scripts, and allocate GPU computation.

Produce and test real edited weight tensors. Do not substitute a research proposal, a purely observational analysis, or a no-edit stopping decision for the required editing experiment.

Never use official test results to modify the selected edit.

Commit all reproducible artifacts and provide the final GitHub path.