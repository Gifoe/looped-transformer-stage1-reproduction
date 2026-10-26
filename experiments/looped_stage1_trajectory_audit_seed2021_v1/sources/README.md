# Training-Free Looped Transformer Reproduction

This repository is an independent reproduction codebase for the inference-time
layer-loop experiments from
[L-z-Chen/Training-Free-Looped-Transformer](https://github.com/L-z-Chen/Training-Free-Looped-Transformer)
and the paper
[Training-Free Looped Transformers](https://arxiv.org/pdf/2605.23872).

The goal is not to mirror the original single-file scripts verbatim. Instead,
the reproduction is split into maintainable modules:

- `src/tflt_repro/strategies.py`: model-independent loop, RK, and anchored Euler strategies with unit tests.
- `src/tflt_repro/patches/qwen.py`: runtime patch for Qwen/Qwen-MoE decoder models, without modifying weights.
- `scripts/run_eval.py`: YAML-driven `lm-evaluation-harness` runner.
- `scripts/reproduce.py`: convenience entry point for the official two-model 16-task reproduction.
- `configs/`: official reproduction config, paper-recipe starter config, and smoke-test config.
- `tests/`: local strategy and config tests that do not download Qwen models.

## Method

Let `g(h)` denote one natural forward pass through the selected loop window.
The public upstream reproduction script for the two Qwen3 runs uses:

```text
natural = g(h_in)
h_iter  = h_in + (1/K) * (natural - h_in)
for t = 1..K-1:
    h_iter = h_iter + (1/K) * (g(h_iter) - h_iter)
h_out = beta * natural + (1 - beta) * h_iter
```

The default official configuration is
`strategy=block_anchored, K=6, beta=0.5, cache_strategy=first`.
This repository also includes a paper-oriented starter recipe for the mid-4
dense/block and MoE/layer settings in `configs/paper_recipe.yaml`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check the environment:

```bash
python scripts/check_env.py
```

Full reproduction requires access to the Qwen3 checkpoints on Hugging Face, a
local GPU, and `lm-eval`. The 30B bf16 weights are roughly 60 GB, so an H100
80GB or similar GPU is recommended.

## Run A Smoke Test

Validate the pipeline with a small number of examples before launching the full
run:

```bash
python scripts/run_eval.py \
  --config configs/smoke.yaml \
  --run q4b_smoke_baseline q4b_smoke_loop \
  --limit 5 \
  --device cuda:0
```

## Reproduce Official 16-Task Runs

Run only the looped configurations:

```bash
python scripts/reproduce.py --models q4b q30b
```

Run both baselines and looped configurations:

```bash
python scripts/reproduce.py --models q4b q30b --baselines
```

Equivalent lower-level entry point:

```bash
python scripts/run_eval.py \
  --config configs/reproduce_official.yaml \
  --run q4b_baseline_16t q4b_blkanc_K6_b050_p1518_16t
```

Outputs are written to `results/<tag>.json` by default. Summarize results with:

```bash
python scripts/summarize_results.py "results/*.json"
```

## Configs

`configs/reproduce_official.yaml` matches the public upstream reproduction:

| tag | model | window | recipe |
| --- | --- | --- | --- |
| `q4b_baseline_16t` | `Qwen/Qwen3-4B-Base` | none | baseline |
| `q4b_blkanc_K6_b050_p1518_16t` | `Qwen/Qwen3-4B-Base` | `[15,16,17,18]` | `block_anchored`, `K=6`, `beta=0.5` |
| `q30b_baseline_16t` | `Qwen/Qwen3-30B-A3B` | none | baseline |
| `q30b_blkanc_K6_b050_p2225_16t` | `Qwen/Qwen3-30B-A3B` | `[22,23,24,25]` | `block_anchored`, `K=6`, `beta=0.5` |

`configs/paper_recipe.yaml` is a starting point for the paper-style recipe.
Task names such as `mmlu_pro` and `gpqa` can differ across `lm-eval` versions,
so check your local task registry with `lm-eval --tasks list` before running a
large sweep.

## Local Tests

These tests do not load any Qwen model:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Implementation Notes

- The patch is a runtime monkey patch. It replaces only the inner model's `forward` method and does not modify weights.
- The default `decode_mode=bypass` matches loglikelihood evaluation: looping happens only during prefill.
- `cache_strategy=first` writes KV cache from the hidden state before entering the loop window, matching the upstream reproduction script.
- For generation tasks, set `decode_mode` to `full` or `first_n`; decode cost increases with K.
