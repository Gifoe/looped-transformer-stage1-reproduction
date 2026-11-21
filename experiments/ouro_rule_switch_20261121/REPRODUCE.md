# Reproduce on the pinned Ouro-1.4B Base checkpoint

Use Python 3.12, PyTorch 2.7.1+cu128, Transformers 4.55.4 and a compatible CUDA device. Set MODEL to the local directory for revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 of ByteDance/Ouro-1.4B. The model.safetensors SHA-256 is 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. All commands use eager shared layers, no cache, BF16 weights exactly promoted to FP32 and no TF32 or autocast.

Run from this experiment's directory in the repository.

```bash
export PYTHONPATH="../ouro_native_state_computation_20261121:../ouro_open_carry_20261121:../ouro_bidirectional_20261121"
export OPENBLAS_NUM_THREADS=4
mkdir -p rerun
pilot=../ouro_open_carry_20261121/MANIFEST.json
boundary=../ouro_carry_boundary_20261121/results/MANIFEST.json
previous=../ouro_bidirectional_20261121/results/DECK.json
python rule.py "$MODEL" "$pilot" "$boundary" "$previous" rerun
python boundary_control.py "$MODEL" "$pilot" "$boundary" "$previous" rerun
```

If the recorded native gate passes, run causal.py and natural.py with MODEL and rerun. To reproduce the natural same-prompt experiment with the frozen discovery-selected local window, run pathway.py between causal.py and natural.py if the graft gate passes. This keeps the discovered window selection before confirmation. For the exact recorded server sequence see the five run scripts. report.py replays the verified analysis when supplied with the server's paths for the earlier results. The result directory contains the locked protocols, model and data verification metadata, full response text, integer parsing, and every generated token ID.

