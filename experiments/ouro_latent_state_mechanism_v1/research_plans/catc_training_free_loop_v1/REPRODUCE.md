# Reproduce the published scalar results

Use Python 3.10 and NumPy 1.26.4. No Torch, model download or GPU is needed for the two statistical replays:

```
python code/reproduce_release.py --release .
python code/reproduce_adaptation.py --release .
```

The scripts verify and assemble any lossless archive shards automatically. The first reconstructs all 1094 training questions and requires exact equality of the fixed-setting summaries, gates, correction norms and capture runtime. The second reconstructs the 299 Joint-Option validation questions and 997 SciQ questions, reproduces their statistics exactly, and verifies all 299 FullAnswer decisions, token sums, confidence intervals and selected-coefficient histories.

CPU replay verifies reported arithmetic and statistical decisions; it does not repeat GPU forwards. Full 49152-dimensional tensors and Q/K/V witnesses remain on the server, identified in RAW_TENSOR_MANIFEST.json. GPU collectors use the frozen external Ouro checkpoint and source hashes, BF16 SDPA and four recurrent passes. For an independent GPU rerun, use a fresh output root, adapt paths/device binding, and run the numerical gates before collecting effects. Preserve the original formal output files.

The executed evaluation source hashes are in the corresponding freeze receipts. A later LoopCD 0.5 helper addition derives a standing baseline from saved native vectors without changing the original completed forward executions or frozen statistical gates.
