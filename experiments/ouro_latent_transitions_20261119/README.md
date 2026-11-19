# Ouro-1.4B K1→K2 transition prediction

This experiment predicts consecutive native post-normalization loop states and tests the predicted response to a real K1 source-token intervention. See the [Chinese report](REPORT_ZH.md), [pre-confirmation protocol](PROTOCOL.md) and [decision record](results/advancement.json).

## Reproduce

- Start from the Git worktree that contains this experiment. The inherited data and capture implementation come from commit 951eb63e448968ee29d91c442def7c4965477e55. Produce the complete corrected parent study, including the projected vectors, by following its [reproduction instructions](../ouro_latent_organization_20261118/README.md). The source manifest and fixed model hashes are verified by the capture job.
- The execution environment used one NVIDIA GeForce RTX 4090 with BF16, Python 3.12.3, CUDA-compatible PyTorch 2.7.1+cu128, Transformers 4.55.4, NumPy 1.26.4, scikit-learn 1.7.2, SciPy 1.15.3, safetensors 0.8.0 and matplotlib 3.9.0. See [environment record](logs/software-versions.txt).
- Download the fixed ByteDance/Ouro-1.4B Base revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. The required SHA-256 of model.safetensors is 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. The [upstream loader](../ouro_cross_task_20261118/run.py) also tests the Git blob hashes of the official config, model implementation and tokenizer files.

Run on the compute server, in the repository root, with a fresh output directory.

```bash
export OURO_PYTHON=/absolute/path/to/venv/bin/python
bash experiments/ouro_latent_transitions_20261119/reproduce.sh \
  /absolute/path/to/corrected-parent-study \
  /absolute/path/to/model-snapshot \
  /absolute/path/to/new-transition-study
```

The GPU phase acquires all four natural worlds per group and the physical interventions in one traversal per task. All transition and encoder fitting uses the natural fit split. Only the discovery split selects configurations. The 96D configuration grid is stored in each discovery.csv. Full-width models refit the frozen latent configurations, with discovery selection for the full-width ridge and context penalties and between the two nonlinear families. The full-width advancement script applies an operational semantic and behavior check that requires positive improvement bounds against the strongest simple comparator. The model missed the geometry and in-domain natural gates independently of this check.

## Records

| Artifact | Contents |
|---|---|
| [runs](runs) | Exact world-to-tensor index, K1/K2 native candidate scores, fixed intervention site, dose, direction, observed clean and edited scores. Earlier K1–K4 scores are marked separately. |
| [results](results) | Seeded configurations, 96D and full-width per-group fit predictions and performance summaries, paired group bootstrap comparisons, and named held-out input results. The reported MSE is computed on the standardized mapping/query/readout roles. |
| [model archives](results/2048) | Trusted local pickle archives with trained encoders and compressed transition parameters. To inspect in a Python process, put this experiment on sys.path and import Scale from fit_study into the main module before loading. The 96D archive includes the direct baselines. Full-width direct baselines are reconstructed from the natural fit split through fit_study.py. |
| [cases](full_cases/cases.json) | Six fixed complete case traces, input text/tokens, semantic contrast, regional effects at all physical and normalization events, plus checksummed array inventory. |
| [figures](results/figures) | Five matched print-vector PDF and high-resolution PNG outputs. Fig. 1 uses group-bootstrap state error intervals rescaled by the identity point estimate. Fig. 3 uses paired group-bootstrap intervals for relative error reduction. |
| [logs](logs) and [earlier attempt logs](initial) | Final runs and preserved failed attempts. The first acquisition stopped when an arbitrary cross-device L2 cutoff rejected an arithmetic calibration sample. The re-run used direction agreement for the historical reference and exact within-device parity for the native boundary. An evaluation run also stopped on JSON serialization of NumPy scalars, and two post-processing runs stopped on NumPy Boolean serialization and a module import collision. These repairs did not change model selection or metrics. |

On the experiment server the original full-width arrays are in /root/autodl-tmp/ouro-latent-transitions-20261119/runs. The six complete clean/edited BF16 token trajectories are under the same output root in full_cases. Their shape and SHA-256 appear in [cases.json](full_cases/cases.json), and the main full-width array shapes follow the indexed JSONL files. The checksum audit can be repeated with summarize_transition.py by passing the original runs and results directories. Reproduction regenerates the arrays without requiring access to the current experiment server.
