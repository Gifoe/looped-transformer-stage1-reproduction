# Ouro-1.4B Base on ARC-Challenge

This study evaluates all 1,172 official ARC-Challenge test questions with whole-answer zero-shot scoring. It compares native K1–K4, published LoopCD variants, a clipped secant readout and its last-step ablation. A separate validation-only intervention studies access to K4 Q, K and V at K2.

- [Results and conclusions in Chinese](REPORT_ZH.md)
- [Benchmark and latency tables](BASELINES.md)
- [Protocol](GOAL.md) · [Literature](LITERATURE.md) · [Experiment log](EXPERIMENT_LOG.md)
- [Per-question predictions, cost traces and exact split/runtime hashes](results/)

## Reproduce

Use a CUDA GPU with BF16 support, Python 3.12, PyTorch 2.7.1+cu128, Transformers 4.55.4, NumPy, SciPy, PyArrow and Matplotlib. Clone the complete research branch so that the sibling runtime and model verification modules are present.

Use the pinned [Ouro-1.4B checkpoint](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1). The loader rejects model/config/tokenizer bytes that differ from this revision.

```bash
hf download ByteDance/Ouro-1.4B --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 --local-dir /absolute/path/to/model
cd experiments/ouro_arc_pareto_20261119
PYTHON=/absolute/path/to/python bash reproduce.sh /absolute/path/to/model /absolute/path/to/arc /absolute/path/to/repro/results
sha256sum -c ARTIFACTS.sha256
```

Place the revision-pinned validation and test parquet files in the data directory if direct Hugging Face access is unavailable. The reproduction script verifies both SHA-256 sums. The artifacts are checksummed for this release. Timing and timestamped metadata vary under reproduction. Output accuracy can vary when software, hardware, batch dimensions or numerical precision change.

## Result files

- `validation_predictions.jsonl`, `test_predictions.jsonl` and `test_verdicts.csv` contain question IDs, official option order, normalized-length denominators or per-method predicted labels. Scores in JSONL are unnormalized cumulative log probabilities.
- `freeze.json` records dev-selected comparisons, screens, source hashes and development data hashes before test scoring. `test_metrics.json` contains paired transitions, resampling intervals and cost statistics. Each fixed-depth method uses the budget in its method name; standalone executions independently verify score identity.
- `oracle_predictions.jsonl` and `oracle_followup_predictions.jsonl` contain distinct subsets of the official validation split, with matched controls. The cached latency command uses explicitly sized `DynamicCache` to support the pinned runtime. The default checkpoint cache fails to initialize on that runtime.
- `adaptive_depth_replay.json` uses candidate-score confidence to choose a K2 or K4 scoring record. It measures a policy in replay. Its reported budget is hypothetical, and the released latency traces correspond to actual fixed-depth execution.

The question and answer texts are available under the upstream dataset terms from AllenAI. This branch publishes derived scores and IDs alongside the code. No model weights are included.
