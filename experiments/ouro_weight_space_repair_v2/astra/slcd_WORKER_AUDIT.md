# SLCD worker source audit

Read-only review of slcd_token_worker.py, slcd_supervise.py and slcd_core.py, plus the initial recipe definitions. No GPU job, validation/test file, score log or model checkpoint was opened. No running source was changed. All three modules parse successfully.

## Actual defects / incomplete receipts

1. **Unconditional parity-success claim.** The worker allows a missing reference file, skips comparison for any ID missing from reference, and permits errors={}. Nevertheless COMPLETE always writes all_baseline_parity_exact=True. Fixed/Adaptive comparisons also depend on recipe IDs tf_fixed/tf_adaptive being present. They are present in the initial config, but the generic worker does not require them. Acceptance must require an exact full reference ID set and eight errors per question (native_k1..6 plus both LoopCD), all zero. Freshly collected selection states are not independently compared against `values` from their generating forward; the external original-score reference is therefore the active parity witness.

2. **Incomplete source/content binding.** The recorded signature hashes the data manifest, recipe file and four source modules, but excludes slcd_core.py, the imported tensor-hash helper and the reference-score file. State files themselves have no recorded/verified content hash. Existing discovery payloads are checked against a matching implementation string, not the previously recorded statistics-file SHA. New flat state payloads bind dataset/prompt/revision but do not bind the generating scorer/runtime implementation. The cache-only branch also bypasses load_model's checkpoint verification, although it records the loaded head hash. Thus complete eight-method zero-error checks can validate observed arithmetic, but these checks alone should not be described as complete source provenance.

3. **Resume state-index incompleteness.** Previously completed rows are skipped before state_index.append, so COMPLETE.states describes only questions newly evaluated in that invocation. On a fully resumed job it is empty. maximum_error also resets to zero instead of being recomputed from existing records. The existing-row acceptance checks only signature and duplicate IDs, not scope/head hash/parity-key completeness. An independent complete index should be reconstructed from all final rows, not from COMPLETE.states.

These issues do not demonstrate numerical corruption of the active job. Keep the running worker hash frozen. A new flat, slcd-prefixed postprocessing audit can gate acceptance by checking all expected IDs/counts, reference and data bindings, exact eight-method zero errors, actual head/checkpoint identity, state content hashes against existing discovery provenance, and new-state generator source provenance. Record current complete code hashes separately. Do not silently repair old receipts to claim checks the worker did not perform.

## Verified code paths and scope

- Cached discovery uses normalized BF16 states, unchanged BF16 vocabulary-head weights, FP32 logits and full-vocabulary log_softmax, with every target position included. It validates six states per candidate and [1, target_length, 2048] shape. F.linear uses the same bias-free head operation as the scorer. Whenever all references exist, nonzero native/Fixed/Adaptive comparison errors abort.
- Selection without --reuse-states loads the original model and a Kmax6 scorer. No edited bundle, parameter copy or weight update appears in this path. Head bytes are checked at exit. Newly written states and results use flat slcd-prefixed files under existing results/. The explicit worker/supervisor/core paths create no new directory; inherited runtime.configure calls mkdir(exist_ok=True), so its directory-preservation property depends on those pre-existing cache directories being present.
- Initial fixed omega=.5 and adaptive omega=1 agree with scorer semantics; gamma0 takes the exact fixed path. Core score_late and trajectory_reference are explicitly score-level methods, distinct from token-level full-logit methods. They must not be labeled as algebraically equivalent to token-logit LoopCD.
- The new ledger/lock are logs/slcd_gpu_budget.json and logs/slcd_gpu.lock. They do not touch the prior V2 budget. Supervisor charges loaded-worker elapsed time, including failures, against21600 seconds; the worker reads its deadline. Completion/error status is explicit for normal child exits and timeouts.
- load_development only accepts discovery/selection. Validation/test branches in the generic worker require lock-file existence, but do not yet verify that their contents authorize the exact recipe file/data signature. This is a real limitation for future held-out use: the eventual controller must compare locked recipe hashes and data authorization, not rely only on file existence. No validation/test files were opened in this audit.

## Reviewed SHA256

| File | SHA256 |
|---|---|
| slcd_core.py | e80c70c7b85e9fa73028038efcc45cf08fe6e07e0219ba5f47e9afcc27b8ae31 |
| slcd_supervise.py | 4f16bc64746319035a9e93928e728fa76b97816b3c5cff492e74ad335c97da1d |
| slcd_token_worker.py | 3e4759152a829bc901a1d450fe881adb12ebfe7d006cfb86c6c069f718f9b1b1 |
