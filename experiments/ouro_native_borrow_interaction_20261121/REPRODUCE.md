# Reproduce the native interaction experiment

Use local ByteDance/Ouro-1.4B Base revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. The expected model.safetensors SHA-256 is 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. Tested on Python 3.12, Torch 2.7.1+cu128, Transformers 4.55.4 and CUDA 12.8. The pinned BF16 checkpoint is cast exactly to FP32, with no cache, TF32 or autocast.

Add the sibling experiment directories for native state, open carry, bidirectional, rule switch, borrow query, borrow transport and borrow factorial to PYTHONPATH. Copy the archived DECK.json to an empty output directory to bypass original server history paths. Run native_deck.py with the local model directory and output directory. Only when the native gate passes run readout.py, mediation.py, trajectory.py and specificity.py, each with the same two arguments. The shell files record original sequencing and content locks. report.py rechecks the full results using the original host layout. The result tree contains exact generated sequences, y/foil pairs, finite-statistic comparators, and the layer-profile plot and its raw data.

