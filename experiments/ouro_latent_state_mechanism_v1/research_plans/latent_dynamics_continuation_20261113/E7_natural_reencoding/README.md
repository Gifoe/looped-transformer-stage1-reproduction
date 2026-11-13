# E7: natural input reencoding and loop-gain controls

Completed 22,280 natural input variants on the fixed Train1094 and exposed Validation299 cohorts. All numerical work and independent arithmetic audits were performed by Sol; Astra directed the frozen design and scientific interpretation. This experiment does not intervene on hidden states.

**Conclusion: inconclusive protocol dependence, not invariance.** The Train symbol/interaction findings did not meet the frozen same-direction Validation confirmation criterion. The intervals also fail the predeclared ±1 percentage-point equivalence criterion. The unique fixed two-view K2 method failed against native K4.

## Primary four-choice results

Effects below are changes in the K1→K2 semantic accuracy gain, not direct K2 accuracy improvements. Every question is one bootstrap block across all16 P/S cells.

| Contrast | Train1092 estimate pp | Train adjusted98.333% interval | Validation295 estimate pp | Validation95% interval |
|---|---:|---|---:|---|
| Query option-content position P | +1.0302 | [-1.3507, +3.3196] | +0.4237 | [-2.9096, +3.7860] |
| Global answer-symbol cycle S | +2.0986 | [+0.3764, +3.7775] | +0.0847 | [-2.7401, +2.8531] |
| P×S interaction | -4.3244 | [-8.6625, -0.0407] | -3.6535 | [-10.3955, +3.1262] |

Three/five-choice extensions are retained separately and are too small for the main inference. Exact ties, semantic-index sensitivity, no-related-tie grids, all offsets, semantic consistency and consistently wrong predictions remain in the complete tables.

The frozen identityK2 + p=floor(m/2),s0,K2 pair scored936/1092 versus native K4964/1092 on four-choice Train, and244/295 versus257/295 on four-choice Validation. No alternate offset, pair or weight was selected. Two K2 forwards do not establish equal latency to one K4 forward.

Gate results are offline readouts after four actual body passes. The new mechanism-first user order superseded all actual pair/gate profile execution; neither has a measured online speed claim. The prepared profile source and explicit NOT_RUN receipt are retained.

## Code and evidence

- [`analysis/`](analysis/): complete Train/Validation reports, per-question/view results, fixed-pair failures, gate baselines and frozen decisions, coverage/runtime and independent raw-unit audit.
- [`source_and_protocol/`](source_and_protocol/): collection, rendering, generator, CPU analysis, source/protocol freezes, technical/semantic/length audits, and the director's final review.
- [`frozen_inputs/`](frozen_inputs/): losslessly compressed exact input JSONL files, including every generated token sequence. Their original byte checksums are in RELEASE_SOURCE_MAP.json.
- [`raw_archives/`](raw_archives/): complete original capture/technical JSON and execution journals in tar.gz, with per-file byte-checksum manifests. No model weights are included.
- [`execution_logs/`](execution_logs/): original logs, losslessly compressed.

Pinned model: ByteDance/Ouro-1.4B revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1; checkpoint SHA25658872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. Runtime: RTX5090, torch2.7.1+cu128, transformers4.55.0, BF16/SDPA/eval/no cache, original25-shot Joint-Option protocol. Same-hardware native endpoint bridges were exact. No Test evaluation or weight fitting was added.

The root independent audit checked every completed unit checksum, semantic mapping, first ties, four-depth predictions, gate readout, complete grid, exact primary means and fixed-pair correctness. It is not a second GPU or bootstrap run. Technical endpoint checks, independent semantic rendering audit, source hashes and original receipts have separate scopes preserved in their files.

P changes query option content positions; S changes demo and query label tokens. Even the query-symbol-preserving diagonal changes demonstration symbols. No result identifies a unique writer, pure label bias, semantic information origin or a complete recurrent mechanism. The subsequent C1 investigation separately tests the validated upstream conditional update's actual L18 residual and attention routes.
