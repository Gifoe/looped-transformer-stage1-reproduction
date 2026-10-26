# Reproduction

This is an independent implementation audit and supplementary benchmark run.
Direct parity against the author Training-Free Loop patch is unavailable because
the author repository returns 404. This limitation is not removed by the HF
identity tests passing.

## Pinned environment

Execution host: Windows 10 18363, NVIDIA RTX 5090 32GB, NVIDIA driver 616.56,
CUDA driver API 13.4, PyTorch 2.11.0+cu128, Transformers 4.57.1, Python 3.13.9.
All model weights are BF16; quantization and LLM training are disabled.
Direct dependencies are pinned in `scripts/prepare_resources.py`; full installed
versions are saved under `logs/`. `lm-evaluation-harness` is not used by this
custom supplementary scorer and its installed version is recorded as absent.

Create an isolated environment with a matching CUDA-enabled PyTorch build.
The server preparer uses the existing PyTorch through system-site-packages and
installs a separate pinned pip NumPy wheel to avoid the base MKL conflict.

## From a clean project

On the evaluated server the project lives at
`D:/looped_stage1_trajectory_audit_seed2021_v1`.
Run in this directory with Python 3.13 and the matching PyTorch installation:

```powershell
E:\Anaconda\python.exe scripts\prepare_resources.py
.venv\Scripts\python.exe scripts\verify_snapshots.py
.venv\Scripts\python.exe scripts\prepare_data.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\ouro_audit.py --stage parity
.venv\Scripts\python.exe scripts\ouro_audit.py --stage benchmark
.venv\Scripts\python.exe scripts\ouro_audit.py --stage pilot
.venv\Scripts\python.exe scripts\ouro_audit.py --stage formal
.venv\Scripts\python.exe scripts\analyze.py
.venv\Scripts\python.exe scripts\qwen_audit.py
.venv\Scripts\python.exe scripts\math_generation.py
.venv\Scripts\python.exe scripts\math_depth_audit.py
.venv\Scripts\python.exe scripts\scoring_and_timing.py
.venv\Scripts\python.exe scripts\rebuild_all.py
.venv\Scripts\python.exe scripts\write_findings.py
.venv\Scripts\python.exe scripts\package_artifacts.py
```

`run_pipeline.py`, `run_extension.py`, and `run_finalization.py` supervise the same stages. The provided
hidden WMI launchers persist when SSH disconnects. Do not launch a second
extension while one is active. GPU stages run sequentially. The budget ledger
includes correctness, failed and resumed GPU processes. Each supervised GPU
process has a timeout equal to the remaining 50-hour allowance, and per-sample
checks stop before the limit. Downloads and installation use CPU; a conservative
minute is reserved for the separate tiny BF16 environment smoke. GPU accounting
is process wall time, an upper-bound proxy rather than an utilization integral.

## Scoring and alignment

- Deterministic seed 2021 and fixed manifests. ARC/OpenBookQA validation and
  official test are disjoint. MMLU is held-out and never selects parameters.
  Dataset sampling uses task-specific deterministic subseeds:
  `2021 + int(sha256(task_or_subject_tag)[:8], 16)`; exact tags and all resulting
  sample IDs/checksums are retained in `prepare_data.py` and the frozen manifests.
- 200 test questions each for ARC-Challenge, OpenBookQA and five MMLU subjects
  in the Qwen comparison. Ouro uses an immutable 100-question subset of each.
- Zero-shot ARC/OpenBookQA scores full candidate text; MMLU scores A/B/C/D.
  No chat template or implicit BOS is inserted. Context truncation is disabled.
- Every candidate is scored from the exact concatenated prompt/answer boundary.
  Summed conditional token log-likelihood is divided by candidate character count
  for ranking. Raw sums and per-token gold NLL are also saved. Softmax over these
  normalized scores yields *choice score weights*, not calibrated answer
  probabilities. Vocabulary entropy/KL are measured separately at the last
  prompt token and never confused with choice-level KL.
- The paper-style `choice_logit_margin` is the top-1 minus top-2 candidate
  normalized score. The separately named `choice_margin` is the probability gap.
  These curves use their own validation threshold grids and are never conflated.
- The checkpoint's released ponder gate is also tested, using its actual BF16
  sigmoid/product/CDF arithmetic and a data-independent alpha grid. Its offline
  sample-level exits are checked against the public model's threshold readout.
- The Ouro states returned after each output norm go directly into the head.
  Hidden LoopCD combines pre-output-norm states, then applies exactly one norm.
  Logit contrast and log-softmax use FP32 after the BF16 head.
- Qwen uses [15,16,17,18], K=2/6, alpha=1/K, and beta=0.5 as configured.
  The likelihood path has use_cache=False. Separate cache tests confirm one
  canonical `first` KV write per layer/token with decode bypass.
- Generation is a separate 100-question GSM8K experiment with three fixed
  training demonstrations, greedy decoding, 256-token cap, and a strict ####
  numeric extractor. Every generated token executes four recurrent loops and
  retains all 96 cache levels. Selected omega=0 variants reuse the mathematically
  identical baseline. An additional omega=0.25 diagnostic uses the smallest
  positive value in the fixed grid; it is not the validation-selected setting
  or the best test-grid setting, and is actually run at every generated token.
  Separate complete answers are also generated at true depths 1/2/3 and with
  nonzero hidden guidance. Different autoregressive prefixes are kept separate
  from the same-prefix four-depth MC trajectory analysis.
  `scoring_and_timing.py` additionally invokes `math_teacherforced_probe.py`
  on the same 100 questions: all four depths score the official reference
  solution, while vocabulary and hidden scalars use the common prompt endpoint.
  These teacher-forced diagnostics and separate generated-answer links are not
  presented as full generated-prefix mechanistic trajectories.

## Rebuild without a GPU

With all raw JSONL files present, install NumPy, pandas, matplotlib and SciPy and
run `python scripts/rebuild_all.py`. This recreates CSV tables and PNG/PDF figures
without loading model weights. Paired bootstrap uses seed 2021 and 10,000 draws;
pattern intervals use Wilson 95% intervals. Strength-grid intervals are pointwise
and unadjusted, so test-grid winners are exploratory. Deployment strengths and
halting operating points are chosen exclusively from validation.

Offline halting reports recurrent work, not real latency reduction. The oracle
reads gold/future and is never considered deployable. Matching the random
depth-assignment histogram checks whether a policy's advantage comes merely
from its average computation.

The model snapshots are downloaded once on the server rather than duplicated in
the local delivery. Exact revisions and LFS SHA-256 checks are in
`results/checkpoint_integrity.json`. Dataset snapshot revisions, file names,
sample IDs and manifest SHA-256 values are in `data/manifest_audit.json`.
