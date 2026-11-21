# Reproduce the frozen borrow-transport experiment

Use the pinned local ByteDance/Ouro-1.4B Base revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. The weights SHA-256 is 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. The tested runtime was Python 3.12, Torch 2.7.1+cu128, Transformers 4.55.4 and CUDA 12.8. The model runs eager with no cache, BF16 weights promoted exactly to FP32, TF32 off, and autocast disabled.

In a clone, put the directories for ouro_native_state_computation, ouro_open_carry, ouro_bidirectional, ouro_rule_switch and ouro_borrow_query on PYTHONPATH using the actual sibling dates in the tree. The history from baseline.py and its imported borrow.INPUTS points to each earlier experiment's exact data manifests in results. To reproduce evaluation without changing these paths, put the released DECK.json in a new output directory first.

Run baseline.py with the model directory and output directory. Run mechanism.py only if the native gate passes. Run readout_check.py only if the matching gate passes. The lock files record protocol and code SHA-256. Artifact verification replays complete-output summaries, tokenizer-based donor selection and cross-phase structural checks using the recorded server layout. The basis files contain the actual orthonormal axes and identifiers used to fit them.

