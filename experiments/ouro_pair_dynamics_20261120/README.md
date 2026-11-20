# Ouro-1.4B ordered pair dynamics

Frozen [Ouro-1.4B Base](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1). This experiment branches from `bfca32210a0d08d6bc03f070f69e94212ffb88f8`. K denotes the actual number of shared transformer passes.

- [中文结果](REPORT_ZH.md) and [claim–evidence table](CLAIM_EVIDENCE_TABLE.md)
- [Theory](THEORY.md), [history audit](BASELINE_AUDIT.md), [prior art](NOVELTY_MATRIX.md), [preregistration](PREREGISTRATION.md), [iteration log](ITERATION_LOG.md), and [paper outline](PAPER_OUTLINE.md)
- [Primary summary](results/SUMMARY.json), [report statistics](results/REPORT_STATS.json), and [figure exports](results/figures)
- [Verification receipt](results/RELEASE_VERIFICATION.json) and [checksums](ARTIFACTS.sha256)

## Pipeline

Use the original snapshot directory including the checked-in model code and tokenizer. The recorded environment includes Python, torch 2.7.1+cu128, Transformers 4.55.4, an RTX 4090, and math SDPA. See `results/ENVIRONMENT.txt` and `results/AUDIT.json` for versions and weight hashes. The baseline repository must include the inherited Ouro experiment directories.

```bash
bash reproduce.sh /path/to/venv/bin/python /path/to/pinned/model /path/to/new/run
python verify_release.py results
sha256sum -c ARTIFACTS.sha256
```

The GPU script reproduces the frozen route and saves logs incrementally. If a task fails, rerun with the same output directory and Python environment. Published JSON and NPZ data are read-only evidence. Use a separate output directory for replay. To regenerate only figures and tables from already verified data, run `python figures.py RUN`, `python report_data.py RUN`, and `python make_report.py RUN REPORT.md`.

## Data layout

`MANIFEST.json` holds inputs, candidate token IDs and role spans for each named cohort. Predictor code delivers only the token IDs, positions and candidate vocabulary indices to the model interface. Scoring is performed after model execution.

`PREDICTIONS_*` and `PREDICTION_SHA_*` contain time-stamped predictions/receipts. `NATIVE_*` contains independently executed outcomes linked by prediction digest. Analogous locks hold for conditional interventions, natural ARC and BF16 replay. Timing fields record synchronous seconds, CUDA peak allocation and operator counts for each batch.

`STATE_AUDIT.json` contains an additional descriptive FP32 comparison at every A/M pair boundary for the first validation group of each family. The `STATE_*` float16 shards retain full token-by-width states of all four edited trajectories at selected boundaries of K2–K4, inclusive of pre/post norm. Development witness shards have an independent first-loop set. Main sample-level mixed-state magnitudes are computed in FP32 or from BF16 states promoted for measurement. All group-level direction sketches in the primary data use a fixed 24-dimensional projection.

Native confirmation includes 192 novel groups. The independent field failure check uses 96 distinct groups. The terminal swap test uses 48 validation groups (24 per family at each of K2 and K4) and shares their native prefix. The ARC option-order analysis and full token witnesses are descriptive supplements to the primary frozen analyses.

