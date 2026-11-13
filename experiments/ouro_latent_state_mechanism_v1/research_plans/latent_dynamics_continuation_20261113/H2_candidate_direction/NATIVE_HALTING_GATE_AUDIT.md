# Ouro-1.4B native halting gate: read-only source audit

Audit by Sol, 2026-11-13. No GPU, model forward, new Test data, raw outcome statistics, fitting, feature search or H2 modification. The baseline code and checkpoint header were inspected; no gate-quality claim is inferred from the existence of its weights. Machine-readable byte/source evidence is in `NATIVE_HALTING_GATE_PROVENANCE.json`.

## Identity and actual checkpoint evidence

The model directory is `/root/autodl-tmp/ouro-e4-e5/model`, snapshot `ByteDance/Ouro-1.4B` revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. The existing verified staging receipt and completed H2 loaded identity both name checkpoint SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`, size 2,869,336,434 bytes. This audit reads the safetensors header and only the gate's byte payloads; it does not rehash the full checkpoint or instantiate another model.

| Checkpoint tensor | Stored dtype | Shape | Payload bytes | Raw payload SHA256 |
|---|---|---|---:|---|
| `model.early_exit_gate.weight` | BF16 | [1, 2048] | 4096 | `fa92644802e800123d4be3d7f18e08fe47e27ac89e6c15a4308cdd568479accc` |
| `model.early_exit_gate.bias` | BF16 | [1] | 2 | `ab2bc9800ff2d8b79cf751d7f6200ffb21b0ee75ae0d91e3017556e25cca8a3c` |

These names and shapes match `OuroModel.__init__`, line513: one shared `nn.Linear(hidden_size,1)` with bias. Gate weights exist in the released checkpoint rather than merely in the class definition. This header check does not prove how they were trained, their useful calibration, their causal utility, or bitwise identity of a newly instantiated in-memory parameter; no fresh weight loading was performed here.

The local `modeling_ouro.py` SHA is `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`. It exactly matches both pinned project copies (`sources/modeling_ouro.py` and `results/model_provenance/Ouro-1.4B__modeling_ouro.py`) and the **actually loaded class-source SHA** recorded by completed H2 capture. Config SHA is `ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3`; configuration module SHA is `950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb`. Full source paths, sizes and hashes are retained in the JSON provenance.

## Exact forward semantics

Source references below are line numbers in that exact `modeling_ouro.py` file.

1. **The backbone always executes its configured body loop count.** Line592 loops `range(self.total_ut_steps)`. Each pass executes all24 physical decoder layers, then final RMSNorm, stores the normalized state, and evaluates the gate on that state (lines606–608). There is no gate-dependent `break` in this loop. With body count4, all4 passes/96 decoder-layer applications occur before the CausalLM wrapper makes its readout choice.
2. **Gate probabilities are stick-breaking hazards per token.** For gate logit g_k, the code computes lambda_k=sigmoid(g_k), p_k=lambda_k times remaining mass for k<4, and updates remaining mass by multiplying (1−lambda_k). The last p_4 is the entire remaining mass; the last gate logit is computed but its sigmoid does not determine p_4. The linear gate parameters are shared across loops. `stacked_exit_pdf` has [batch, token, loop] dimensions (lines721–733). There is no explicit FP32 conversion for sigmoid/product/CDF in this path; with BF16 gates its numerical arithmetic must be preserved for parity.
3. **Inference strategy precedence is forced valid step → threshold → weighted logits → last state.** `exit_at_step=None` falls back to `self.early_exit_step`; `exit_threshold=None` falls back to `self.early_exit_threshold` (lines689–695). Then lines775–805 select a valid indexed step, else threshold, else weighted logits, else the final state. An invalid indexed step silently falls through rather than raising an error.
4. **The actual released config has `total_ut_steps=4`, `early_exit_threshold=1.0`, and no `early_exit_step` field.** Thus the unmodified default call with `exit_at_step=None` reaches the **threshold1.0 per-token CDF readout**, not a special bypass of the gate. The mathematically intended final cumulative mass is1, but BF16 products/sums can hit the representable threshold earlier or miss it until fallback. This audit did not evaluate any examples and therefore does **not** claim default readout and forcedK4 are bitwise identical on this dataset.
5. **Forced native K4 is explicit index3 with body count4.** `model(..., use_cache=False, exit_at_step=3)` chooses the fourth normalized hidden state and a single `lm_head` call. The default gate/CDF tensors are still computed, but do not affect the selected hidden state. Setting only `exit_at_step=1` would choose the second state's logits after a four-loop body; it does not save body work. Physical fixed-depth2 additionally requires setting the body's actual loop count to2. If the body count were2, index3 would be out of range and fall through.
6. **Threshold readout is token-wise, not one depth for the whole question.** Lines781–800 take the first CDF crossing separately at every sequence position, fall back to the last available loop if none crosses, gather corresponding states, and apply the head. A last-prompt-token/sample-level policy is a different explicitly defined evaluation protocol, especially for teacher-forced multi-token answers.
7. **Weighted exit mixes raw logits.** `compute_expected_logits` evaluates a head for each loop, multiplies logits by per-token exit probabilities cast to the head dtype, and sums them (lines737–756). This is a weighted raw-logit readout, not mixture-of-answer probabilities. With the stock threshold1.0 present, merely passing `use_weighted_exit=True` does not activate this branch: threshold has priority. Both the step attribute and threshold attribute must be disabled/set toNone to reach weighted inference. This entails four bodies' recurrent passes and four heads, not a compute-saving halting procedure.
8. **The labels path differs from ordinary inference.** Whenever `labels is not None`, lines762–772 unconditionally request the weighted-logit readout and call `self.loss_function`; forced-step/threshold strategy selection is in the labels-absent branch. This code permits training gradients through the gate-weighted readout. It does not establish the actual released training schedule, a ponder penalty, useful learned stopping, or a gate-quality result.

`use_cache` defaults to true in config, but the project comparison explicitly uses `use_cache=False`. Cache capacity and recurrent indexing are separate concerns; the source audit does not extend its conclusions to adaptive cached generation.

## Existing project evaluation was found

There is already a native gate baseline in pinned project source. Treating it as entirely unevaluated would be incorrect.

| Existing artifact, relative to `experiments/looped_stage1_trajectory_audit_seed2021_v1/` | What the inspected source establishes | Limitation |
|---|---|---|
| `scripts/native_gate_probe.py`, `gate_cdf` and `run` | Extracts the actual gate logits at the final prompt token; preserves BF16 sigmoid/product/CDF; two data-independent text probes compare thresholds .25/.5/.75 against public threshold logits using exact tensor equality; exports a sample-level gate readout. | Offline full-body extraction; not real recurrent-time savings. This audit inspected code, not raw evaluation rows, and did not rerun it. |
| `REPRODUCE.md`, lines75–78 | Explicitly documents released ponder gate, actual BF16 arithmetic, data-independent threshold grid and readout parity. | Project documentation, not an official model-training recipe. |
| `scripts/analyze.py` | Includes `native_gate_cdf` among the existing validation-selected halting comparators when its cached field is present. | Historical selection/evaluation, not a new5090 benchmark or permission to inspect/reselect on Test. |
| `extensions/baseline_suite_20261025_seed2021/scripts/actual_halting.py`, `adaptive_first_candidate` and `score` | Hooks the live norm/gate, accumulates prefix-only gate CDF, and raises a controlled exception to omit later loops. Last prompt prediction position of fixed candidate0 chooses one question-level depth; other candidate bodies use that same depth. Decoder-layer call counts are checked. | This is sample-level candidate0-prefix halting, not official per-token gathered readout. Its `main` explicitly opens historical Test paths; **do not run that entrypoint in the current stage**. Only the implementation can be reused under a newly frozen allowed-input runner. |
| Extension `scripts/policies.py` and `scripts/audit_actual_halting.py` | Existing BF16-threshold handling and actual-halting audit implementation are available for source reuse. | Their old protocols, datasets, thresholds, model versions and hardware do not establish current-run efficacy. No historical result is silently ported. |

This audit does not quote/recompute historical accuracy tables or load historical/new Test examples. The existence of implementation and project documentation is established; historical empirical validity is not re-audited here.

## Directly reproducible baseline definitions

These are existing readout/policy definitions, not a third feature-search family. No command below is executed by this audit.

| Baseline | Definition that must be frozen explicitly | Actual work |
|---|---|---|
| Native fixedK4 | Set body/config total loops4, explicit `exit_at_step=3`, no labels, no cache. | Four body passes, one final head; gate computed but ignored for choice. |
| Stock default readout | Same body4, `early_exit_step=None`, released threshold1.0, `exit_at_step=None`, no labels. | Four passes; per-token gate/CDF gather followed by one head. It is not a runtime-early-exit baseline. |
| Stock threshold readout | Same body4, `early_exit_step=None`, predeclared threshold tau, no labels. | Four passes; token-wise first crossing. Threshold tuning, if authorized later, must obey the current Train-only/freeze policy. |
| Released weighted-logit readout | `early_exit_step=None`, `early_exit_threshold=None`, `exit_at_step=None`, `use_weighted_exit=True`, no labels. | Four passes/four full-shaped heads. Preserve shape and BF16 arithmetic for an exact native comparison. |
| True sample-level gate early exit | Reuse existing gate-hook/CDF logic on fixed R of a frozen allowed prompt; stop the body at the first frozen threshold crossing; last depth fallback. | Only completed loops plus gate/head work. Requires fresh native endpoint parity, layer-count and total-timing checks under the allowed protocol; cannot be claimed from readout alone. |

For candidate-logit parity, preserve full head shape and candidate/input identity as in the existing5090 technical gate. A saved scalar CDF, a reread of forced-depth scores, or choosing a depth after all4 loops is not evidence of genuine runtime savings. An unmodified default call and a forcedK4 call should be labeled separately until direct endpoint parity is checked under the actual allowed input protocol.

## Missing evidence and defensible conclusion

The downloaded official checkpoint artifacts contain exported config and forward code, but the model directory and pinned provenance bundle inspected here do not include an official model card or full training recipe. Consequently this audit can establish a stored gate, its exact exported readout/loss graph, and prior project evaluation implementations. It cannot establish the gate's original training objective/schedule, whether it was successfully optimized for adaptive computation, calibration at any threshold, or whether it improves current accuracy/compute. No conclusion of “trained effective gate” follows from a nonmissing tensor or a differentiable labels path.

The gate is an existing reproducible comparator whose default exported inference does **not** implement computational early termination. There is prior project code for a genuine prefix-stopping variant. It can be reconsidered under a separately authorized frozen evaluation, without inventing another latent feature block or modifying H2. Which comparison to run next remains the root/Astra scientific-direction decision; this audit launches none.
