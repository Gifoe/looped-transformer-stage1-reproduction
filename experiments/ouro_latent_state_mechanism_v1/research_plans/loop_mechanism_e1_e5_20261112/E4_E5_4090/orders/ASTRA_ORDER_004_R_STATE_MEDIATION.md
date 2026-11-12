# Astra Order 004 — one-site state-to-attention mediation

Execute the following single-site discriminator. Query-factorial Validation is complete: Q-only rollback loses 10 native corrections, joint QKV loses 18 of 104, and 86 survive even whole L18 attention-output rollback. Peer E2 state64 already tests full/probe-subspace L18 R-state rollback, but does not independently restore the mediator under that block; its ordinary restore resets x2 itself. The proposed four corners are therefore not a duplicate. No scan across layers, heads or token regions is authorized.

## State and mediator

Receiver is natural K2 at physical L18/index17, actual loop1. R is the final input token and must be asserted. X2 and X1 are the same-question true L18 layer-input R residuals, from current-hardware native K2 and K1. U replaces only X2_R by X1_R before L18, leaving every non-R state unchanged. Perform a real complete forward under U and capture its actual SDPA_R, named A(U). Natural K2 SDPA_R is A2. Do not substitute natural K1 SDPA output A1 for A(U): under U all non-R KV still come from K2.

Four cells, each running L18 projection/norm/residual/MLP and L19–L24 normally:

- N: X2 with A2 (native).
- U: X1 with its actual A(U) (upstream state block).
- U_restore: X1 with A2 explicitly restored at SDPA_R.
- A_only: X2 with A(U) injected at SDPA_R.

The required mediator restoration changes only SDPA_R while retaining the actual upstream block. It is not an identity undo of X1→X2. This establishes a controlled state/attention path contrast, not how the earlier recurrent computation originally generated X2.

## Controls and provenance

Use one complete same-hardware native K2 baseline with source/input/full-logit/state bridge to saved results, and one explicit identity reinjecting X2 and A2. Reuse saved K1 X1 only after tensor/source identity verification; do not rerun the original 26 conditions.

Input-state controls, with natural downstream attention: X2 rescaled to ||X1||, and X2 plus an unrelated question's same-site X1−X2 direction scaled to the target ||X1−X2||. Mediator-restore controls under the same U block: A(U) rescaled to ||A2||, and A(U) plus an unrelated question's A2−A(U) direction scaled to the target ||A2−A(U)||. Freeze both donor maps by lexicographically sorted IDs before outcomes; record zero-norm fallbacks/unavailable cases. Norm-only controls have the known local-normalization limitation and are not a universal scale explanation test.

Capture actual U Q_R/K_R/V_R, A(U), normalized attention increment, R residual after attention and after MLP, and final-normalized R state where feasible. Compare hashes for non-R input/Q/K/V/SDPA against native, and assert causal masks/current_ut/full-shape SDPA/no-cache unchanged. Both node hooks must be scoped to the intended repeated-layer call. Source identity, on-disk versus actually loaded model code, input tokens and candidate mapping remain fixed. Save all raw candidate logits/scores, state vectors, controls and source hashes in a new namespace.

## Cohort and checks

Same frozen train64, then all 299 Validation questions with the unchanged comparison. These are exposed cohorts, not blind holdout. First two fixed IDs test new composed hooks and exact identity/old-baseline bridge. Repeated U forwards must reconstruct identical A(U); U_restore and A_only must receive the intended exact row. Donors are selected without current gold. Keep source modules separate from completed producers.

The eight frozen precision IDs receive a scoped FP32-SDPA/BF16-writeback comparison for A(U), with source/precision scope explicit; if infeasible report it rather than calling previous Q sensitivity an independent test of this branch. Record actual cost and forward-window memory scope. No model weight change or final-state/gold-logit injection.

## Frozen contrasts and claims

For each of meanwrong, recipient-fixedwrong, global-K1-fixedwrong and dynamic margin, report:

`total = F(U) - F(N)`

`attention_only = F(A_only) - F(N)`

`direct_at_native_attention = F(U_restore) - F(N)`

`restore_effect = F(U_restore) - F(U)`

`interaction = F(U) - F(U_restore) - F(A_only) + F(N)`.

Preserve exact raw-logit signs, top ties, natural four groups, all opposite-direction rescues/damages and all paired cases. If U damages a native correction, distinguish mediator restoration recovery, persistence of damage after restoration, and controls that also recover or damage it. Do not turn the effect contrast into a mediated percentage. If U_restore remains harmful relative to N, this supports a path through the modified residual and later computation under the stated intervention; it does not locate the original generating layer, establish global necessity, or exclude compatibility/OOD effects. If A2 restores the decision, report that conditional mediation evidence with all controls. Complete the design even when effects contradict the working explanation.
