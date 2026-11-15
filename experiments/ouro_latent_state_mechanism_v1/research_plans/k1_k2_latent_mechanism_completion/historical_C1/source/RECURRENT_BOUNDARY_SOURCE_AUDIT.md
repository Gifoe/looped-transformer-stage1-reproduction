# Actual recurrent boundary and L18 branch semantics

Root Sol read the pinned `modeling_ouro.py` source copied in `D:/ouro-e7-20261113/`, with expected SHA256 `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`. This is a source audit, not a new GPU result.

## Boundary

The body initializes hidden states once from token embeddings (line585). It executes the same 24 physical layers at each `current_ut` (lines592-604), then replaces the hidden states with `self.norm(hidden_states)` (line606). That normalized full-sequence tensor is appended to the readout list and supplied to the halting head (lines607-608); it remains the next loop's input. Token embeddings are not added again at the boundary. Position embeddings are prepared once before the recurrent loop. With no cache, no earlier-loop KV cache is reused. A final K1 readout state at R alone therefore does not specify the complete K2 input: the entire sequence recurs.

Actual recurrence: K1 L24 output -> shared final RMSNorm -> full-sequence K2 L1 input -> layers1-24 -> shared final RMSNorm -> readout. A boundary intervention must specify whether it acts before or after the shared final RMSNorm. A decoder lens on one R vector does not test availability or native use of information stored elsewhere in the prompt.

## L18 branches

Each decoder layer saves its incoming tensor as `residual` (line401), applies input RMSNorm, full attention and output RMSNorm (lines402-414), then forms the actual BF16 sum `hidden_states = residual + hidden_states` (line415). Define incoming residual x and the normalized attention increment a at this addition. The model then saves that same x+a tensor as the MLP skip (`residual`, line418), normalizes it, computes and normalizes the MLP increment, and adds the skip (lines419-422).

To test residual-bypass restoration while retaining corrupted attention, attention must first run with the corrupted actual input. The intervention must then form the explicit BF16 sum of the selected residual and selected attention branches before the MLP. Substituting the layer's full incoming x also changes attention, so it is a different intervention. Altering a by a floating correction for x is not guaranteed equivalent to selecting the two branches and performing the native BF16 addition.

A `post_attention_layernorm` prehook that returns a replacement tensor can change its normalized input while leaving the already-saved MLP skip pointing at the original tensor. It does not implement the intended branch restoration. A same-storage in-place change at the boundary or a source-locked layer wrapper needs independent technical acceptance proving both the MLP input and its skip receive the intended x+a tensor, with unchanged non-R rows and full suffix recomputation.

These facts constrain the experiment; they do not establish which branch causes a correction, where answer-relevant information originated, or whether the K1-to-K2 improvement is a single mechanism.
