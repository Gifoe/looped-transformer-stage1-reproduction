# Reproduction notes

Use a fresh directory for a new run. Do not overwrite the supplied evidence. Scripts resolve ROOT from their own location. The reference run uses Linux aarch64, Python3.12.3, GB10, Torch2.11.0+cu130 and Transformers4.55.4; exact package versions are captured in logs/ENVIRONMENT.json. Different hardware/backends need new parity checks.

1. Copy scripts/, sources/, configs/protocol.json and the source notices into the fresh root. Supply a compatible installed CUDA Torch environment via OURO_SHARED_ENV; bootstrap creates a new environment and references those existing packages read-only. It refuses to replace an existing env.

```bash
OURO_SHARED_ENV=/path/to/existing/cuda/env python3 scripts/bootstrap_env.py
env/bin/python scripts/download_model.py
env/bin/python scripts/download_arc_split.py --split train
env/bin/python scripts/build_train_manifests.py
env/bin/python scripts/restart_original_pipeline.py
```

2. The last command charges strict original-call readiness,192-question discovery and the bounded causal screen through supervise.py. It runs shared-invocation confirmation only for qualified modules. Study files and logs are immutable resume inputs; completion is required before scientific interpretation. Read the entire screen, including regressions, before making a new decision.

3. The provided STAGE_C_DECISION, frozen_baseline_recipe and weight_edit_manifest record this run's no-edit decision, if present. They are evidence, not permission to skip causal gates in a new research run. For exact frozen-baseline replay, copy those fixed definitions, then run:

```bash
env/bin/python scripts/run_no_edit_evaluation.py
env/bin/python scripts/assemble_no_edit_report.py
```

The evaluation script freezes candidate definitions, runs192 held-out training questions, then official299validation. It writes FINAL_LOCK before downloading or constructing all1172test rows. Missing locks, changed recipe/data/code hashes and partial cohorts cause failure. It does not select a test coefficient or depth. The three chord coefficients were fixed in advance; none is chosen as a primary method.

All GPU workers must use supervise.py and the independent28800-second ledger. Existing study outputs must be resumed only with identical hashes, or retained while a new study name/directory is used. setup/download/analysis are CPU tasks; the ledger conservatively charges the entire loaded scoring child, including loading and failures. No gradients or training are used.

Important source detail: formal whole-sequence scoring MUST call model.model(input_ids=x,use_cache=False) without an explicit attention mask. The mathematically equivalent all-ones mask changes HF4.55.4 SDPA/GQA dispatch and changed some BF16 answers in the engineering audit. Cached scoring was also rejected; it is retained only as a clearly unadmitted diagnostic implementation.

The full-candidate character-length normalization, original choice strings, fixed25demonstrations, all official choice counts, BF16 head then FP32log-softmax, and fixed/adaptive LoopCD formulas are part of the protocol. Do not substitute next-letter prediction, token-count normalization, a1165four-option subset, or historical varying-demo scores.
