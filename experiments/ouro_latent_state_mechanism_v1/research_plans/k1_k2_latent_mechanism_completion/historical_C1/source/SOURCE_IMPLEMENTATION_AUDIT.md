# C1 execution foundation: source and schema audit

Sol read-only audit, followed by CPU hook-framework preparation authorized by the frozen C1 order. No C1 GPU/model forward, new Test evaluation or weight modification occurred during this audit. Actual-profile execution remains paused; its prepared source/fixture are preserved. Scientific operator/dose/direction/cohort are prescribed by `ASTRA_ORDER_C1_UPSTREAM_TO_L18.md`, not selected by this implementation audit.

## Exact residual computation and the alias hazard

Audited checkpoint source is the `modeling_ouro.py` whose SHA256 is `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`. The saved copy is `D:/ouro-e7-20261113/modeling_ouro.py`; runtime source must again attest to this byte identity.

At lines401–415 the physical decoder saves its input as `residual`, computes preLN attention, applies o_proj followed by `input_layernorm_2`, and performs `hidden_states = residual + hidden_states`. At lines418–422 it assigns **that same postadd y object** to the MLP residual before calling `post_attention_layernorm`, MLP, `post_attention_layernorm_2`, and the final residual add.

Therefore `x`, raw SDPA head-space `O`, projected attention output, normalized increment `a`, postadd `y`, and postMLP state are distinct nodes. A new tensor returned from the post-attention norm prehook changes its normalized MLP input but leaves the caller's saved MLP skip pointing at the old y. Returning a compensated a+(xN−xU) is also invalid: BF16 reassociation does not implement the required branchwise add.

The minimal source-locked implementation is to let actual U attention finish, record its x/QKV/O/a, calculate the prescribed actual-dtype sum x_branch+a_branch, and **mutate only the existing y tensor's R row in place** at `post_attention_layernorm` entry, returning None. Both aliases then see that row. Clone actual operands/y before the mutation; save used operands and the actual postMLP state. Root's independent explicit-layer wrapper must verify real skip/y/complete suffix endpoints; the producer's same-storage flag is not an independent audit. Root's BF16 toy already demonstrates that clone-return and in-place composition differ, but the toy is not a real-model identity gate.

## Reuse points in existing E4/E5 source

Paths below are rooted at `D:/ouro-e4-e5-20261112/` and are read without changing them.

| File and lines | Audited implementation | Reuse boundary |
|---|---|---|
| `exec/runner.py:22–39` | Physical depth setting, explicit exit index, input-token assertion, full-shaped public forward, BF16 R logits → FP32 full-vocabulary log_softmax. | Reuse scoring/execution semantics. Old CLI ROOT/environment/lock are not the current5090 scheduler. |
| `exec/observer.py:19–57` | L18 index17 scoped by current_ut; captures true layer input x_R, raw SDPA O_R, normalized a_R and final norm R. SDPA is evaluated at full shape; any replacement affects R only. Exact dtype/non-R guards and try/finally restore are present. | Appropriate observer design. Its `patch` is **SDPA O**, not normalized a or residual bypass. Do not pass a_R to this patch parameter. |
| `exec/state_mediation_components.py:9–55` | Input prehook can replace x_R before attention, captures actual Q/K/V/O, post-attention y and postMLP R; nested SDPA wrapper delegates to the previous function. | Observations/hook-scoping are reusable. The x replacement changes QKV and is the old E4 intervention, not C1 residual-bypass restoration. |
| `exec/state_mediation_runner.py:13–16,31–45,47–81` | Compounds old input block and independent SDPA mediator patch, validates actual A(U), non-R and shapes, archives reference/intervention vectors and exact full-logit bridges. | Reuse provenance, serialization and complete-forward practices; do not relabel these old corners as the new upstream-to-branch experiment. |
| `exec/depth_state_mediation_runner.py:16–18,32–65` | Parameterized receiver/depth donor; true K1 is separately retained despite X1/A1 predecessor aliases. | C1 is fixed K2; do not inherit late-loop aliases as true K1. |
| `exec/confirmation_state_mediation_runner.py:14–17,32–77` | Same staged collect/replay and exact current/donor state checks for fixed confirmation inputs. | Separate prior dataset/precision selection; does not define C1's approved full1094/299 cohort. |
| `exec/attested_runner.py:9–24` | Hashes actually returned model/config modules, not merely files placed beside the checkpoint. | Carry this attestation into the new loader. Do not import its monkeypatching wrapper as an uncontrolled global side effect. |

Order004 explicitly requires A(U) from the actual perturbed receiver, not natural K1 attention; order005 preserves that rule at K3/K4; order006 preserves all old controls and separates globalK1 from predecessor aliases. C1's true U is the supported upstream subtraction, not old natural-K1 L18 input rollback. Whole preattention x restoration can reset the causal suffix and is only an identity anchor here, not the new bypass intervention.

## Exact E3 conditional-update operator

`branch_reference/capture_branch_reference.py:7–37` captures actual local branch outputs. `reference_deltas` at lines39–44 computes N.double()−P.double(), not natural K2−K1. `e3_conditional_factor_hook.py:7–14` implements FP64 current-site proposal plus/minus a fixed FP64 reference, followed by one BF16 cast, recording actual before/after, effective delta and cast residual. Its hook at lines16–43 selects physical indices9..15, current_ut1 and `input_layernorm_2`; non-R output is cloned unchanged. The actual current site's value can already reflect upstream interventions at earlier selected layers; it is not clamped to a saved P vector.

`e3_conditional_factor_question_v2.py:12–30,33–64` combines the unchanged N/P Q9 background with this fixed operator; the old `N_minus_L_main` is the relevant signed subtraction. `e3_conditional_l1030_bare.py:32–50` independently replaces saved actual postnorm attention rows and checks all endpoint/target hashes. These sources establish the wiring and arithmetic; their old4090 tensors cannot serve as exact5090 references.

The original Q region is the audited prefix boundary through R exclusive, not a newly parsed stem. `producer_observer(... layer8 ...)` applies the same-question K1 layer9-input Q region only during current_ut1, preserving prefix/R. New compact caching can retain that Q slice alone; a refactored caller must not substitute a zero-padded full tensor into an old full-hidden donor API without proving identical region writes.

## Existing raw schema inspected

The old Train technical `.native.json` has `id/split/R/targets/input_ids_sha256/metadata_sha256/identity`, native depth rows, actual upstream row, N/U site receipts, SDPA argument descriptions, prior JSON/tensor hashes and tensor hash. N/U receipts include original/applied x_R, Q/K/V_R, actual SDPA_R, non-R hashes, after-attention and after-MLP rows. The `.results.json` has a `rows` list, percondition raw candidate logits/FP32 scores/BF16 sensitivity/full-R-logit hash, intervention tensor hash and explicit unavailable controls. This audit inspected keys/conditions, not new outcome statistics.

New C1 schema is separately fixed: `references/<split>/<IDsha>.json/.safetensors`, with `conditions[K1,N,P,U]`, `tensor_sha256`, full BF16 R logits and same-hardware site references; `treatments/<split>/<IDsha>/<condition>.json/.safetensors` with top-level score row and tensor `logits`. Actual shared N/U operands are losslessly deduplicated through exact recorded hashes and reference-file binding rather than copied as full sequences into every condition.

## BF16 and runtime traps

- L18 O_R is [1,16,128]; x_R/a_R are [1,2048]. Equal element count does not make these interchangeable sites.
- `input_layernorm_2` is after projection and local RMSNorm; zeroing/replacing raw SDPA is a different operation from subtracting the fixed reference from actual a_R.
- Physical layer indices and current_ut must both be scoped: K2 is current_ut1, not “second hook invocation globally”. Repeated-context stacks must end empty; all wrappers restore globals in finally.
- A globally wrapped SDPA function must delegate outside its explicit active layer/UT. Nested L9/L18 wrappers must preserve the previous callable and restore in reverse order. Do not recompute only a query row or alter shape/mask/strides/precision silently.
- Preserve model weight versions. The new activation in-place write is intentional; parameter writes are forbidden. Disable cache and training/dropout; preserve the exact explicit native head shape.
- FP64 control scaling and one BF16 writeback are part of the operation. Norm-only is not equal-dose control. Zero requested delta is a valid identity; nonzero requested change with undefined donor direction is unavailable, not an invented fallback.
- Old cached4090/A6000 predictions and states are source evidence only. NewN/P/d/U must be fresh5090 and bridge current native K1/K2 hashes. Use the current cu128/sm120 runtime and the shared E6 GPU lock, not old producer-relative environments/locks.

## Storage contract

The available filesystem was reported around6.6GB before C1. CPU prepare computes a byte-level capacity estimate using all fixed1393 input Q lengths and the exact retained tensor schema; it does not choose samples. Full per-layer/full-sequence or full-QKV archives would exceed capacity and are not required. Retain compact Q9 K1-Q donor slice, per-site actual before/after/reference/cast residual, actual L18 branch/reference tensors and percondition used branches/y/postMLP/finalnorm plus full R endpoint logits. Full QKV tensors remain unarchived; R rows and complete argument/non-R hashes bind their actual computation. Check current free bytes against the emitted estimate before launching each approved stage; no cleanup of historical evidence is authorized here.
