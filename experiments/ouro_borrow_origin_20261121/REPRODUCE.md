# Reproduce the K1 write and K2 component test

The reference stack is Python 3.12, Torch 2.7.1+cu128, Transformers 4.55.4 and CUDA 12.8 on one NVIDIA RTX 4090. Verify the pinned ByteDance/Ouro-1.4B Base revision and SHA-256 recorded in results/VERIFICATION.json. Run with the checkpoint locally present.

The prior published experiment directories listed in the snapshot shell scripts supply the pinned no-cache eager model runtime, the exact parser, matching helpers and historical prompt ledgers. Provide those directories via PYTHONPATH. For a standalone evaluation copy the archived DECK.json into the results directory, then run origin_deck.py with the pinned model path and results path as positional arguments. Run origin.py with the same paths if the native behavioral gate passes. Full deck regeneration and provenance checks in report.py require the historical datasets used by the parent. Check source locks, reported result hashes and ARTIFACTS.sha256.

Each intervention occurs only at the common pre-digit answer pivot. Subsequent token generation runs the model's original suffix. The normalized MLP writing point is the output of post_attention_layernorm_2 at physical layer 15. Results include target-clustered paired intervals, intentional wrong-answer reversions, same-question rescues, geometry controls and aligned-coverage failures.
