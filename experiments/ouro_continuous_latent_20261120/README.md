# Ouro-1.4B continuous latent dynamics

[中文研究报告](REPORT_ZH.md) · [实验协议](STEP_SIZE_PROTOCOL.md) · [数值对齐](results/NATIVE_PARITY.json) · [权重和 GPU 验证](results/GPU_TIMING.json)

| Question | Results | Figure |
| --- | --- | --- |
| What happens inside native K2? | [Trajectory analysis](CONTINUOUS_DYNAMICS.md) | [1. Native residual updates](results/figures/figure_1_native_residuals.pdf) |
| Do artificial microsteps approach stable dynamics? | [Dynamics and full ARC validation](CONTINUOUS_DYNAMICS.md) | [2. Paths by resolution](results/figures/figure_2_dynamics_by_step_size.pdf) · [3. Resolution convergence](results/figures/figure_3_convergence.pdf) |
| Does the change have semantic structure? | [Grounded probes and baselines](SEMANTIC_TRAJECTORIES.md) | [4. Semantic readout](results/figures/figure_4_semantic_readout.pdf) · [5. Computed paths and interpolation](results/figures/figure_5_computed_vs_interpolation.pdf) |
| Does a candidate semantic state affect the answer? | [Matched-suffix interventions](CAUSAL_EVENTS.md) | [6. Dose-matched edits](results/figures/figure_6_causal_edits.pdf) |
| What happens on native rescues and regressions? | [ARC accuracy and strata](CONTINUOUS_DYNAMICS.md) | [7. ARC transition strata](results/figures/figure_7_arc_transition_strata.pdf) |

## Reproduction

Clone the whole branch so that the inherited experiment modules and 3×3 dataset are available. Use Python 3.11 or later, PyTorch 2.7.1 with CUDA 12.8, Transformers 4.55.4 and the Python packages listed in [environment.txt](environment.txt). The source runtime uses PyTorch SDPA with BF16 tensors and a 24 GiB RTX 4090.

Download the files from [Ouro-1.4B revision 574fa66](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1) and the validation parquet from [Ai2 ARC-Challenge revision 1664417](https://huggingface.co/datasets/allenai/ai2_arc/tree/1664417/ARC-Challenge). The exact checkpoint and dataset hashes are in [COMPLETE.json](results/COMPLETE.json) and [FREEZE.json](results/FREEZE.json).

```bash
cd experiments/ouro_continuous_latent_20261120
bash reproduce.sh /path/to/model /path/to/validation.parquet /path/to/new/run/results
sha256sum -c ARTIFACTS.sha256
```

The checksum command verifies the published artifacts in a clone. The reproduction command writes a new run, including machine-specific timing and reconstructed data tables.

The release contains one JSONL row per measured world, full-token scalar distances and update statistics, candidate logits, diagnostic probe predictions, the full paired causal scores and dose receipts, and full-width token trajectories for one witness from each structured family. The computation creates full-token tensors in memory. The witness selection in [study.py](study.py) controls which tensors are written.

ARC text is supplied by the original Ai2 dataset under CC BY-SA 4.0. Processed token IDs preserve source example IDs and answer metadata. The study uses the validation split.
