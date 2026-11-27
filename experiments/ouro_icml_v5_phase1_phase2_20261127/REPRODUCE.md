# Reproduce and audit Ouro ICML v5

## Audit the released results

From this experiment directory, run:

    python verify_release.py

This verifies the release SHA-256 inventory, checks the 40 official prediction files after decompression, checks the original prediction receipts, checks that question IDs and answer keys match across methods, and recomputes the published per-task scores and the eight-task equal-weight mean. Inputs use the exact K4 historical zero-shot candidate scoring and MMLU-Pro free generation protocol. The release preserves the original scorer receipts and choice text where emitted by that scorer.

## Run the experiment

Original Linux runner paths are kept in archived code and signed receipts. Set up the inherited environment and resolve its data and model paths before launching. Batch wrappers take original study paths as configured in the source. The public snapshot includes V4 utility code under dependencies/v4_snapshot and uses earlier repository branches for the common Ouro hooks, data preparation, historical evaluators, and exact benchmark requests.

- Base and official model setup: frozen Ouro-1.4B Base model.safetensors SHA-256 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. The inherited runtime used 24 GiB GPUs, Python 3.12, PyTorch 2.7.1+cu128 and Transformers 4.55.4.
- Archived dependencies: research/ouro-functional-controllability-v3 at 4e99a6ced59a83d4beb6d1c34dc49b9f2f1be860, research/ouro-loop-aware-functional-identification, research/ouro-mixed-full-benchmark, research/ouro-external-data-joint-training, research/ouro-minimal-functional-feedback, and research/cross-model-functional-controllability.
- Strict Track A: with the V4 environment loaded, run codex_supervision/v5_prose_corpus.py then codex_supervision/strict_1a_archive/v5_prose_preflight.py. The archive is the version that produced the signed strict result. Its original root location is recorded in source. Preserve the FAIL branch and the audit documents. The WikiText development follow-up used p1_gpu0_csbp_and_e0_centered.py and is tagged exploratory.
- Track B: use codex_supervision/v5_matched_peft.py and the memory-recovery wrappers with the exact sources, data-order hashes and seeds in track_b_baselines/PROTOCOL.json. Training adapters used 840 updates and effective batch 4. Memory recovery split logical batches or performed separate candidate forwards and applied the same softmax-loss gradient. Use codex_supervision/v5_peft_official.py for the seal, freeze, evaluate and aggregate commands, then codex_supervision/v5_phase1_close.py for the gate summary.
- Supplementary Phase 1 official evaluations: p1_v5_official_winners.py used official_v5_winners/SELECTION_V5_WINNERS.json and its SHA-256 signature. The result table and per-benchmark comparisons are under official_v5_winners/official.
- Four exploratory Phase 2 modules: p2_gpu1_cross_model_cfit_and_qwen.py. MODULE1 through MODULE4 identify source conditions and development metrics. Checkpoints are included where the runner wrote them.

The pretrained model blobs, training corpora, source prompt banks and optimizer-resume states are kept outside Git. Official predictions, published result tables, retained learned controllers, code, and integrity manifests are included. Historical absolute paths in receipts describe the original execution and may require remapping for training reproduction.

