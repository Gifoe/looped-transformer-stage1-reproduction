# Run the reproduced experiment

Use Python 3.12, PyTorch 2.7.1+cu128, Transformers 4.55.4, CUDA 12.8 and the local pinned directory for ByteDance/Ouro-1.4B revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. SHA256 of model.safetensors is 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. Execution is eager no-cache FP32 from the stored BF16 weights with both TF32 and autocast disabled.

From a repository clone add the neighboring experiment directories ouro_native_state_computation_20261121, ouro_open_carry_20261121, ouro_bidirectional_20261121 and ouro_rule_switch_20261121 to PYTHONPATH. The history paths in borrow.INPUTS map directly to MANIFEST.json from ouro_open_carry, results/MANIFEST.json from ouro_carry_boundary, results/DECK.json from ouro_bidirectional, and results/DECK.json and results/CONTROL_DECK.json from ouro_rule_switch. The archived DECK.json can be placed in a clean output directory to reproduce generated behavior without reinitializing those paths. The server run scripts show all arguments and sequencing.

Run borrow.py with the model directory and output directory. If the native gate passes, run interchange.py and context.py with the same arguments. The other commands run conditionally as described by their own protocols. report.py replays the artifact verifier on the original server paths. The released directory includes all exact inputs, selected target traces, response text, and generated token IDs.

