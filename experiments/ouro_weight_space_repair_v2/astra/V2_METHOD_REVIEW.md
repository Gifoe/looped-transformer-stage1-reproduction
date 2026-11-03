# V2: MIT-style empirical weight editing, bounded implementation review

Status: V2 protocol/utility review, before any V2 candidate results. Read the new user task in full. V1's No-Go is historical context and is NOT a gate for this experiment. V2 must actually evaluate both A sparse-MLP and B low-rank real-weight candidates, choose exactly one edited checkpoint on official validation even if inferior, and evaluate that checkpoint both normally and with its own-state LoopCD. No validation/test outcomes were read for this review. All new files and CPU smoke artifacts are inside V2.

## What MIT actually did

The p50c primary worknote specifies `W = W0 - 6*outer(u, mask64*(u@W0))`, with distinct unit directions and 64 selected MLP output columns at each of two layers. The directions came from matched productive/repetitive contexts; the record includes dose, layer-isolation and random-control checks. This is sparse **direction subtraction**, not simply multiplying selected columns by a scalar. Alpha 6 and layer IDs 15/19 are specific to that model/task and should not be copied into Ouro. [Primary worknote](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/runs/p50c/work_notes/notes.md)

The final answer attributes the improvement largely to recovery of previously truncated practice generations. That failure regime differs from Ouro's fixed-answer teacher-forced scoring, where no generated thinking sequence hits a length limit. This limits expected transfer; it does not justify skipping the requested actual weight tests. [Primary final choice](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/runs/p50c/final_answer.md)

The original manifest records two `[1024,3584]` matrices and explicitly selected columns. Its support count is not automatically the actual BF16 changed-coordinate count. V2 therefore computes changes after materialization and saves exact evaluated targets. [Primary edit manifest](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/runs/p50c/work_notes/frozen_weights_manifest.json)

## Forward collection and deterministic statistics

Use the planned 384-question training discovery pool (original discovery192 plus192 previously unused training examples), excluding fixed25 demonstrations and the separate selection192. Do not consult prior test/validation scores. Cache no model states across candidate edits. Use the verified original whole-sequence call and a single K6 trajectory per candidate continuation to obtain K1–K6 and own-state LoopCD.

Collect all24 MLP down_proj and all24 attention o_proj sites at loops3/4/5/6. Per question/site/loop store:

- `y[i,s,k]`: mean raw projection output before branch RMSNorm, averaged over target-prediction positions within each answer candidate, then equally across choices.
- `a2[i,s,k,j]`: analogous mean squared input to that output projection. MLP dimension5632 and attention dimension2048 must be checked from actual tensor shapes.
- Complete native candidate scores/margins and normalized boundary states at the prediction positions.

Compute sums/means in FP32 and store means, not whole-prefix activation tensors. Keep question weight independent of answer length and number of continuation tokens. These arrays summarize activations; they are not learned probes. They ignore token-specific cancellations and, for `a2`, activation sign; report these limitations.

The following definitions use only these arrays and original physical weights, and can be frozen before any edited inference:

1. For transitions t=(3→4,4→5,5→6), compute each question's native gold-margin change `g[i,t]` and raw-output difference `dy[i,s,t]=y[i,s,k]-y[i,s,k-1]`. Select the bottom and top quarters of g separately at each t. With384 questions, these groups contain96 questions each. Resolve ties by SHA256 of `V2-trend|2021|transition|question_id`; do not select groups only from correct-to-wrong transitions.
2. Give t weights `w=(0.50,0.25,0.25)`. This uses all three transitions while emphasizing formation of K4, the primary selection depth. Define `D[s]=sum_t w[t]*(mean_bottom dy - mean_top dy)`. This is an explicit label-conditioned forward mean contrast permitted by V2, not optimizer-based fitting.
3. Rank sites by `S[s]=||D[s]|| / (sqrt(sum_t w[t]*mean_all_i ||dy[i,s,t]||²)+1e-12)`. Rank MLP and attention separately; tie-break by physical layer index. This dimensionless statistic identifies contrast concentration, not a proven harmful mechanism. Select the top two MLP sites for A, and the top MLP plus top attention site for B. Every candidate initially changes one physical layer; an optional composite below changes two.
4. Rank1 basis: `u=D/||D||`. Do not guess an arbitrary direction if the contrast is numerically zero; choose the next nondegenerate site in the frozen ranking. Fix vector sign by making its largest-magnitude coordinate positive, for reproducible factor files; sign does not affect the projector.
5. Rank4 basis: retain u and add three PCs of within-bottom-group centered dy. For each transition form `dy_bottom - mean_bottom(dy)`, project these rows orthogonal to u, multiply by `sqrt(w[t]/96)`, and stack all three transitions. Use deterministic CPU SVD and take its first three right singular vectors. QR-orthonormalize `[u,PC1,PC2,PC3]`, fix signs, and save the resulting exact FP32 U. Reject a numerically rank-deficient basis instead of disguising random filler as a measured direction; rank1 still satisfies family B. Record singular values and the basis construction.

No supervised loss minimization, backward operation, trained adapter or learned controller appears in these definitions. Nevertheless labels influence the group means, so do not describe this recipe as label-blind. The direction is an empirical candidate, not established causal attribution.

## Sparse MLP column ranking compatible with the collected arrays

Only squared input moments are available, so do not pretend to compute signed neuron-writing means. For each selected MLP site define:

`c[j] = sum_t w[t]*(mean_bottom(a2_k-a2_prev)[j] - mean_top(a2_k-a2_prev)[j])`.

Use `score[j] = max(c[j],0) * (u.T @ W[:,j])²`, and select Top32/64/128 columns, with integer column index as tie-break. The product estimates loss-associated increase in directional writing energy. It is a heuristic; branch normalization prevents interpreting it as an additive answer effect. If all scores are zero, use the predeclared fallback `mean_all_i,k a2[j] * ||W[:,j]||²`, explicitly labeled unsigned-energy fallback. Do not improvise a new scoring rule after seeing candidate accuracy.

Family A primarily removes the rank1 component only within the selected columns, `W[:,S] -= alpha*u*(u.T@W[:,S])`, at alpha1. This follows the structure of MIT's edit more closely while avoiding its model-specific alpha6. One explicitly frozen whole-column attenuation candidate at alpha.5 distinguishes this choice from suppressing all directions written by those columns. The complete14-candidate list below replaces the earlier12-candidate proposal before any edited feedback.

## Bounded candidate and selection plan

Freeze fourteen actual edits before candidate evaluation:

| Family | Sites | Counts/ranks | Strengths | Number |
|---|---|---|---|---:|
| A: sparse rank1 directional subtraction | top2 MLP sites | Top32/64/128 | alpha=1 | 6 |
| A: selected-column attenuation control | top MLP site | Top64 | alpha=.5 | 1 |
| A: fixed two-layer sparse rank1 subtraction | both selected MLP sites | Top32 at each | alpha=1 at each | 1 |
| B: dense direction projection | top MLP, top attention | rank1 | alpha=.25/.5 | 4 |
| B: dense direction projection | same two sites | rank4 | alpha=.25 | 2 |

These modest projection strengths change direction without copying MIT's aggressive alpha6. Ouro applies RMSNorm after the output projection, so uniform whole-matrix scaling would be mostly canceled. Column attenuation and directional projection generally change the direction of the normalized residual write; actual BF16 behavior must be measured rather than inferred from delta norm.

Screen all14 on the same fixed64 training discovery questions; this guarantees real evaluation of both required families. Promote best2 per family (4 total) to selection192, then best1 per family plus the best remaining candidate (3 total) to official validation299. This family-preserving policy is frozen before any edited-candidate feedback. Rank candidates by K4 acc_norm correct count, then K4 mean gold margin, then fewer actual changed BF16 coordinates, then fixed candidate ID. Record K3/K5/K6 but do not change this ranking rule after seeing them. Baseline is a reference row, never eligible for final edited-checkpoint selection. Final validation picks exactly one of the three edited candidates even if all underperform baseline.

There is no adaptive composite branch: the two-layer Top32 edit is already part of the fixed14. No extra layers, counts, strengths or combinations may be introduced from validation/test feedback. Each edited candidate changes at most two physical layers.

Edited+Fixed LoopCD (omega=.5,K4) is the primary combined model. Adaptive cap1 is secondary. Both must use K1/K4 states from the SAME selected edited checkpoint's own trajectory. No original-state substitution, test-driven coefficient tuning or best-test-depth selection is allowed.

## Utility delivered: scripts/weight_edits.py

- `sparse_column_attenuation(name,W,columns,alpha,provenance)` constructs family A.
- `low_rank_projection(name,W,U,alpha,columns=None,provenance)` constructs family B and optionally its explicitly labeled sparse directional variant.
- `applied_edits(model,edits,invalidate=scorer.invalidate)` copies real BF16 tensors into physical parameters, verifies edited hashes, and restores exact original clones on success or exception. It requires eval mode and `requires_grad_(False)`. All recurrent invocations use the changed parameter, without an activation-only substitute.
- `summarize_edits` reports bitwise and numerical changed counts, intended support, BF16 actual Frobenius/max norms, whole-model fraction and raw tensor hashes. A no-change candidate is rejected. Embeddings, tokenizer, head and normalization weights are outside the whitelist. At most two distinct physical layers may change in a candidate.
- `save_edit_bundle` saves the full exact BF16 target matrix for every edit, a sparse-replacement or dense-target exact patch, and small FP32 U factors where present. `load_edit_bundle` validates original bases, file hashes, full-target hashes, factors and changed counts. Factors document construction; exact target/patch bytes determine the evaluated checkpoint, avoiding a second GEMM/rounding path.

CPU synthetic tests passed for real-parameter application of A and rank4 B, rank1 materialization, protected-head preservation, exception restoration, cache-invalidation callbacks, exact patch reconstruction, and full-BF16/factor persistence. Synthetic changed counts were16 for A,64 for rank4 B,249 for rank1 B; these are utility checks, not Ouro experimental results. Root must still perform a small real-checkpoint BF16 apply/score/restore/reapply gate before candidate screening.

## Budget and interpretation

The additional6 GPU-worker-hour cap includes loaded-worker time, model copies, scoring and failures. Reuse a single K6 trajectory for each method's own depth/readout scores; do not claim separate per-depth latency measurements. At approximately the V1 whole-question throughput, discovery384, candidate screen896 question-edits, selection768, validation897 and two final1172-question model trajectories leave useful margin under21600 seconds. Measure V2 throughput and preserve the full final-evaluation reserve rather than assuming all CPU materialization time is free.

This is an empirical editing experiment. A negative result is valid only after actually testing both families, selecting one edited checkpoint, and completing its frozen final evaluations. Positive benchmark changes do not automatically prove a novel shared failure mechanism. V1's stopping argument must not be used to avoid V2's required real edits.
