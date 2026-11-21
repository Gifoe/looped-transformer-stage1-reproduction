# Reproduce a factorial state-interchange run

Use the pinned local ByteDance/Ouro-1.4B Base revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 and model.safetensors SHA-256 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. The tested runtime is Python 3.12, Torch 2.7.1+cu128, Transformers 4.55.4 and CUDA 12.8. The model uses native eager no-cache execution, BF16 weights promoted exactly to FP32, and no TF32 or autocast.

In a clone put the sibling Ouro native-state, open-carry, bidirectional, rule-switch, borrow-query and borrow-transport experiment folders on PYTHONPATH. To avoid rebuilding server-specific history paths, copy the released DECK.json into a clean output directory. Run factorial.py with the local model directory and the output directory. If the native gates pass, run graft.py and natural.py with the same arguments. Run interaction.py only when the forward four-corner gate passes. The four shell runners record the server ordering and exact source directories. The artifact verifier recomputes the arithmetic, overlap checks, full-integer outcomes, gates and clustered intervals on the original host.

