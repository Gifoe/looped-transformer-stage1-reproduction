# Reproduce iteration 1

Use a full checkout of the research branch, the revision-pinned Ouro-1.4B Base model, one CUDA GPU with BF16 support and the revision-pinned official Train parquet files. The measured run used PyTorch 2.7.1+cu128, Transformers 4.55.4, PyArrow, Safetensors, NumPy and SciPy. The loader verifies the official model checkpoint, tokenizer and modeling source bytes. [Input provenance](DATA_PROVENANCE.md)

Run each stage from this experiment directory. Use an empty output directory for a new reproduction. The commands below can also be run with the single stage `all`.

```bash
MODEL_DIR=/absolute/model-directory
ARC_TRAIN=/absolute/arc-challenge-train.parquet
OPENBOOK_TRAIN=/absolute/openbookqa-main-train.parquet
RUN_DIR=/absolute/new-run-directory

bash reproduce.sh train-dev "$MODEL_DIR" "$ARC_TRAIN" "$OPENBOOK_TRAIN" "$RUN_DIR"
bash reproduce.sh postgate-eval "$MODEL_DIR" "$ARC_TRAIN" "$OPENBOOK_TRAIN" "$RUN_DIR"
bash reproduce.sh postgate-cost "$MODEL_DIR" "$ARC_TRAIN" "$OPENBOOK_TRAIN" "$RUN_DIR"
```

`train-dev` audits native K6, trains all four arms and evaluates ARC development. `postgate-eval` requires a passing development receipt and the same adapter checksums. It evaluates the sealed OpenBookQA IDs once, then evaluates ARC diagnostics and runs the same-model and mixed-loop shams on ARC development. `postgate-cost` waits for a GPU with an empty compute list, then makes 4,736 synchronized batch-one measurements across 64 prompts, two interleaved repeats and 37 method-depth configurations. Leave the selected GPU exclusive during this stage.

The transfer runner refuses an already opened output namespace, and the cost runner preserves existing raw timing rows. The source [execution freeze](EXECUTION_FREEZE.json) and [post-gate freeze](POSTGATE_FREEZE.json) record the evaluated code hashes. Published [training and score records](iteration1/) contain the original experiment's per-question scores, checkpoint receipts and audited timing.
