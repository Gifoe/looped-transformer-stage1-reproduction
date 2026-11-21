# Reproduction on the pinned model

Run from this experiment directory in a clone of the release. Use Python 3.12, PyTorch 2.7.1+cu128, Transformers 4.55.4 and CUDA 12.8 as recorded in the result metadata. The checkpoint is ByteDance/Ouro-1.4B revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. Use its local directory for MODEL. The model.safetensors SHA-256 is 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. The pinned checkpoint is loaded as bfloat16 and exactly promoted to float32. Disable CUDA TF32 and autocast as in the shared setup.

```bash
export PYTHONPATH="../ouro_native_state_computation_20261121:../ouro_open_carry_20261121"
export OPENBLAS_NUM_THREADS=4
mkdir -p rerun
python bidi.py "$MODEL" ../ouro_open_carry_20261121/MANIFEST.json ../ouro_carry_boundary_20261121/results/MANIFEST.json rerun
python operand.py "$MODEL" rerun
```

Run state.py only when the independent common-answer gate in NATIVE_SUMMARY.json passes. Run channels.py only when the independent operand gate in OPERAND_SUMMARY.json passes. Both take MODEL and an output directory as positional arguments. Raw sequence token IDs and text are under results/. The server job launchers record the order of the stage gates. The release verifier checks data exclusions, prompt ancestry, raw-to-summary reconstruction and intervention replay.

