# Causal computation atlas

This atlas corresponds to ByteDance/Ouro-1.4B Base revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. All numeric contents come from original BF16 native recurrent forwards or named interventions on those forwards. See [the mechanism report](../HIDDEN_COMPUTATION.md) and [the frozen analysis](../MECHANISM_FREEZE.md).

## Array layout

- Each group/protocol `.npz` corresponds to one 3×3 semantic tableau. `role_*` arrays have shape `[event, token_role, contrast]`. They aggregate full 2,048-dimensional residuals and the full native width for projections and updates. They retain 96 events ordered K1–K4 × zero-based L00–L23, with four separate events in `role_loop_norm`. Missing roles are zero because they have no assigned positions.
- `token_*` arrays for the first three groups of each split have shape `[event, token_position, contrast]`. They preserve the per-token full-width RMS. Companion `*_causal.npz` arrays contain paired differences in the source cut, noise cut, sham, restore, and matched re-cut runs. Contrast definitions are stored alongside each array. They preserve masks on the same input and sequence length.
- `lens` arrays contain the original head at each physical layer on the last token after the model's tied final normalization. Shape is `[loop, layer, world, target_channel, answer_index]`. World order is source-major, then rule index. Target channels are candidate content and candidate label. This is a fixed head diagnostic at an internal site.
- Candidate probabilities and unique top-choice values use the original full-vocabulary head on K1–K4 loop outputs. They are in [the natural rows](../results/mechanism/natural.jsonl); [intervention rows](../results/mechanism/causal.jsonl) include arm, recipient group, source/rule index, and intervention loop.
- `fork_atlas.npz` compares native fork states for matching question tokens, final answer positions and unchanged candidate content slots. The last axis identifies the state RMS values and internal tied-head contrast. NaN marks absent roles or an uncomputed diagnostic.
- `ATLAS_INDEX.csv` records group, split, protocol, bytes and sha256 per raw array file. Source IDs, factor values, assignment and exact vocabulary are in [the released manifest](../results/mechanism/mechanism_tableaux.jsonl). The natural and fork rows also include exact token IDs and role masks.

## Full width tensor archive

[Raw layout](../results/mechanism/raw_layout.json) records a binary capture of the two fully aligned native content-list worlds (S,M)=(0,0) and (0,1) in the first development group. It includes every input and output residual, the Q/K/V learned projections before rotary transforms, attention branch output and actual normalized attention update, MLP branch output and actual normalized MLP update, and each loop boundary. Every raw BF16 coordinate and full input position is preserved. The binary is held on the execution host because it exceeds the Git single-file size cap. The layout and capture script allow deterministic regeneration. Set `OURO_RAW_ATLAS` to a fresh local path with space for the binary before loading the model.

Input and output states of the shared prefix were checked bitwise at every physical layer and loop norm, in one batch with common padded length. Causal masks use the same explicit causal SDPA backend for all sham and cut conditions. The branch retains the [native endpoint gate](../results/mechanism/native_gate.json) and [per-group shared-prefix parity](../results/mechanism/prefix_parity.json).
