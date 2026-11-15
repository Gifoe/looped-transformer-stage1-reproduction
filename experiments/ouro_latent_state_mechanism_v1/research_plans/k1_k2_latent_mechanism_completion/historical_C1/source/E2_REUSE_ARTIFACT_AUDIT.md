# E2 frozen decoder and state reuse audit

2026-11-14. Current Sol performed a read-only source, schema and artifact-availability audit. No fitting, model inference, GPU work, new Test outcomes, or new performance statistics were performed. Existing evaluator source was inspected, never imported or executed. C1 and its frozen FP32 source were unchanged.

**Current readable artifacts are missing. E2 cannot presently be described as directly loadable/reusable.** Sources and historical hash receipts survive; the actual selected coefficients/scalers, pooled probe-state file and span NPZ were not located in the audited local publication/handoff trees or current server. Re-fitting would create a new decoder and is not a substitute for recovering the frozen one.

## Exact artifacts and availability

Original E2 root is `/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E2/`.

| Relative artifact | Historical expected SHA256 | Source-defined contents |
|---|---|---|
| `probe_fit_attempt3/selected_models.npz` | `204a3788301c7aa826a198b9e40ce495ef81284b6cd41a4758dfd942f8650890` | 16 models, each `w[2049,5]`, `mu[2048]`, `std[2048]`; FP64; 48 arrays |
| `probe_fit_attempt3/train_val_selection.json` | `e7b8823c6c29358077f357340252f298955b2087764b83fd89cf7a571fbe6759` | Selection metadata and selected regularization; filtered local receipt survives |
| `E2_fixed_probe_states.safetensors` | `8ccae86df9786e88fa63219d10c9ba101ae96f49793da7f63ae01449e49e62e5` | Four BF16 R vectors per question, each `[1,2048]` |
| `E2_latent_transition_rows.jsonl.gz` | `5d749790c56ab7bf2de3a9a64609e0a5b383fe3b7c5b2ec0cc5d5129586daac4` | Source-linked question metadata; not read here |
| `state_intervention/technical_train_probe_span.npz` | `76302d62517710db248b58e209657943d87a563137a1a9d5b609b738525dc657` | FP64 basis `[2048,4]`, singular values, tolerance; receipt explicitly says technical-only, not scientific target freeze |

These hashes are **historical expected hashes**, not newly verified hashes of absent files. The accompanying `E2_REUSE_ARTIFACT_INDEX.json` gives actual hashes of 12 readable sources/receipts and separates them from expected artifact hashes.

Inventory: exact primary filenames were absent from the local `D:/ouro-e1e2-publish-20261113` and `D:/ouro-e4-e5-20261112` trees, including member-name inspection of 38 ZIP/tar.gz archives. Current server recursive `/root/autodl-tmp` search found none of selected models, original selection JSON, fixed probe states or span NPZ. The latent-transition rows were absent from the local E1_E2_A6000 release; their current-server basename was not separately checked. Public release contains extraction/fit/evaluator/subspace sources and receipts, but no usable frozen coefficient/state payload was found. This is a bounded inventory, not proof of deletion from the original host or all backups.

Original A6000 transport is identified by AST inspection of `D:/ouro-e4-e5-20261112/exec/transfer.py:4`: `1.tcp.eu.cpolar.io:10145`, user `luansiyu`. No connection was attempted. Old seeta connection metadata points to `connect.bjb1.seetacloud.com:40321`, user root, for E4/E5 transport; that alone does not identify it as the original A6000. No credentials are included in either audit artifact.

## Frozen decoder contract

Authoritative final fit is release `source/E2/e2_fit_probe_v3.py`, SHA `ca24cef4aab75616d3267b9a4479ea508e2fc4e8b8e78d119bc06d3983e3652c`. Earlier versions are not substitutes: v3 penalizes bias as well as feature weights.

The 16 keys follow `{l18_input|post_norm}_{raw|unit_l2}_trainK{1|2}_{real|permuted}`. Each key has `_w`, `_mu`, `_std` arrays. The Train fit uses 1094 non-demonstration questions; 25 Train demonstrations are excluded. Regularization is chosen using the existing 299-question Val set, which had prior exposure and is not a pristine new validation sample. Ordinary masked multinomial cross-entropy plus `lambda/2 * sum(w**2)` is used, not class-balanced ridge. Five fixed penalties and source-loop Val logloss choose a penalty; exact selection ties prefer stronger regularization. Permutation control is conditional on exact offered-target tuple, not a new random target vocabulary.

Feature loading is `feat(rr,site,u)` (v3 lines 28–30), with `u=0` meaning K1 and `u=1` meaning K2. BF16 states are converted through FP32 to FP64. For `unit_l2`, normalize each example first, mapping zero vectors to zero. For each site/type separately, pooled K1+K2 non-demo **Train-only** states determine coordinate mean and population standard deviation; zero std becomes 1 (lines 54–62). The same transform is reused by both source-loop fits and controls. Val and Test do not fit the scaler.

Prediction-only formula, conditional on recovering and hash-verifying the original NPZ:

```python
# x: one original-coordinate 2048-vector; frozen model key n
phi = x.astype(np.float64)
if feature_type == 'unit_l2':
    phi = phi / np.linalg.norm(phi) if np.linalg.norm(phi) else phi * 0
z = np.r_[(phi - npz[n+'_mu']) / npz[n+'_std'], 1.] @ npz[n+'_w']
# Map offered token IDs to global columns; mask others to -inf.
# Softmax only offered classes. Report scores in the original offered order.
# Exact score ties use first offered position, not sorted global token ID.
```

This is a documented API, not a newly executed inference function. Original fit/evaluator scripts have top-level execution and must not be imported for convenience. Final frozen evaluator `D:/ouro-e4-e5-20261112/handoff/e2_frozen_test_once.py` independently checks weights, selection and source hashes before applying coefficients; it must not be run in this task.

Global column mapping is token-ID sorted: `0:330=ĠA`, `1:340=ĠC`, `2:389=ĠB`, `3:414=ĠE`, `4:422=ĠD`. Mapping was verified by reading the current original-model tokenizer vocabulary, without model execution. These are answer-label tokens, not semantic candidate IDs or correctness classes. A row with any unknown offered class is unsupported. Source-fit K1 and source-fit K2 are different trained heads; applying one fixed head to both loops is the meaningful same-decoder comparison.

## Geometry and intervention contract

`raw` is affine-linear in the original residual coordinates. `unit_l2` adds per-example nonlinear normalization, then coordinatewise centering/scaling: it is not an angle-preserving spherical decoder, and raw/unit comparisons cannot be called pure linear angle effects.

`D:/ouro-e4-e5-20261112/state_intervention/e2_state_intervention.py:12–19` defines raw score-contrast directions as `(w[:-1,1:] - w[:-1,:1]) / std[:,None]`, excluding bias; FP64 SVD supplies an orthonormal basis. The mean cancels in contrasts. Projection is `B @ (B.T @ delta)` in original residual coordinates. This raw-coordinate span formula does not turn a unit-normalized probe into a fixed globally linear map.

The saved technical span uses `l18_input_raw_trainK1_real`. Global five-class contrasts have rank 4, including the extra E axis. An ordinary four-offered-label contrast span has at most rank 3. The independent note records the rare E-offered class with no E gold in Train; treating global rank 4 as exactly the offered four-choice subspace is incorrect. This audit selects neither a new rank nor a new decoder.

The interceptor asserts physical layer index 17, loop index 0 or 1 (intervention source lines 57–65): `l18_input` is the R residual at L18 pre-forward, **before pre-attention normalization**, not post-attention activation and not Q itself. Replacing it changes both the residual path and local attention Q/K/V; it is not query-only manipulation. `post_norm` is the model's final normalized loop state at R. Site, loop, model source, hardware and input hashes must remain explicit in any later reuse.

## Tensor scope and independent audit limits

Extraction source lines 31–40 reads original natural branch files from `/home/luansiyu/loop-latent/research_plans/natural_branch_full_run_20261111/{split}_{sha256(id)}.safetensors`. All per-layer `x,a,y,b,z` are explicitly BF16 `[1,2048]` R-only, across both loops and all 24 physical layers. It checks actual BF16 additions and layer chaining. It checks **R-only** `u1_layer00_x == loop1_post_norm`. Four extracted vectors per question are keyed `{split}_{sha256(id)}_u{0|1}_{l18_input|post_norm}`. The historical extraction covers 2590 questions, including Train/Val and historical Test; no historical Test arrays were opened here.

Thus the surviving receipts attest a historical R-only chain, not a retained full-sequence loop boundary. Neither the full-cohort original R branch payloads nor the pooled R-state payload were located in the audited current inventory. Existing E4/E5 R-native snapshots are different runs/sites and cannot silently replace E2 frozen states. The technical E2 driver can save `native{1,2}_input_full` at **L18 input** for three selected technical questions; this is not a full-cohort K1-postnorm→K2-input boundary archive. Its source capability does not prove that those technical tensor files are currently readable.

`e2_posthoc_audit.py` separately hash-locks the catalog, coverage, geometry source, config, rows and state file, and historically checked all four probe vectors against each original branch R tensor. Its receipt survives. It explicitly states that this post-hoc check cannot prove the collector checked hashes before collection. `E2_V3_INDEPENDENT_AUDIT.md` distinguishes independent toy objective/gradient checks from a production fit rerun and documents the v3 correction history. None of those receipts recovers absent coefficients or full-sequence tensors.

For Experiment A, the immediately reusable material is the frozen specification, source and expected-hash contract. Actual reuse requires recovering the original NPZ and any required state payload, checking their exact hashes and schema, and respecting old A6000 versus current5090 state provenance. This audit authorizes no new fitting, capture, intervention, or scientific choice.
