# Ouro-1.4B hidden computation

This study uses the frozen ByteDance Ouro-1.4B Base checkpoint. It pairs protocol comparisons with natural forward passes and interchanges at specified loop, layer, and token positions. The previous research commit is `5557699b244a0a6981beeb540dc58491bf165096`.

- [Chinese report](REPORT_ZH.md) and [mechanism findings](HIDDEN_COMPUTATION.md)
- [Causal computation atlas](CAUSAL_COMPUTATION_ATLAS/) and [protocol and LoopCD comparisons](PROTOCOL_COMPARISON.md)
- [Mechanism-to-performance assessment](MECHANISM_TO_PERFORMANCE.md) and [handoff](MECHANISM_HANDOFF.md)
- [Prior experiments](PRIOR_AUDIT.md), [literature](LITERATURE.md), and [input identities](input_manifest.json)

## Reproduction

Use Python 3.12, PyTorch 2.7.1+cu128, Transformers 4.55.4, NumPy, PyArrow, SciPy, and Matplotlib on a CUDA device with BF16 support. Take the repository from the published commit and include the sibling `ouro_cross_task_20261118`, `ouro_latent_state_mechanism_v1`, and `ouro_arc_pareto_20261119` code. The checkpoint loader verifies weights and the tokenizer/config source before inference.

```bash
hf download ByteDance/Ouro-1.4B --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 --local-dir /absolute/path/to/model
python fetch_inputs.py /absolute/path/to/data
```

Run the native protocol grid and mechanism confirmation before scoring LoopCD. Use a fresh output location with about 1 GB for full-width tensor capture and arrays.

```bash
PY=/absolute/path/to/python
MODEL=/absolute/path/to/model
DATA=/absolute/path/to/data
OUT=/absolute/path/to/reproduction
mkdir -p "$OUT"
PYTHON="$PY" bash reproduce_protocol.sh "$MODEL" "$DATA/arc_challenge" "$OUT/protocol" native
bash reproduce_mechanism.sh "$PY" "$MODEL" "$DATA/hotpot_distractor_validation.parquet" "$OUT/mechanism"
PYTHON="$PY" bash reproduce_protocol.sh "$MODEL" "$DATA/arc_challenge" "$OUT/protocol" loopcd
PYTHON="$PY" bash reproduce_protocol.sh "$MODEL" "$DATA/arc_challenge" "$OUT/protocol" costs
"$PY" verify_release.py "$OUT/protocol" "$OUT/mechanism/atlas" "$OUT/mechanism/hotpot" --hotpot-hard "$OUT/mechanism/hard"
```

The benchmark downloader accepts `--mirror` and verifies the same pinned SHA-256 after download. You can also add `--historical-test` to verify the ARC test parquet. The archived A6000 trajectories and reference demonstrations used in the protocol report are linked in [the prior evidence audit](PRIOR_AUDIT.md). The current reproduction evaluates new questions on ARC Train/Validation only.

To verify the files in the published commit, run `sha256sum -c ARTIFACTS.sha256` and `python verify_release.py protocol_results results/mechanism results/mechanism` from this folder.

The upstream [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) and [HotpotQA](https://hotpotqa.github.io/) dataset terms apply to the source material. This branch stores derived measurements, controlled task fixtures, provenance and code. Checkpoint weights and full upstream parquet files remain outside Git.
