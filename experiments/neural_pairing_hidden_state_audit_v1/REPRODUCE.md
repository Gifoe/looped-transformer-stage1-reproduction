# Reproduce the audit

Use CURRENT_SCOPE.md as the authoritative task scope. Training preparation is retained but must not be launched as part of these instructions.

## Recompute from saved evidence (CPU)

For the two delivery archives, extract the raw-data tar into an empty experiment directory first, then unzip the code/results archive into the same directory. Final derived analyses and plots in the code/results package take precedence over the earlier raw-data snapshot. Raw batch JSON and state tensors are unchanged.

The artifact includes the immutable published JSON, fresh ARC batch JSON, study selection manifests, causal batch JSON, selected-position state tensors, results and code. Model weights are separately pinned in configs/model_transfer.json; they are not redistributed. Python3.12 with numpy/scipy/torch/matplotlib is sufficient for the following analyses. Use UTF-8 on Windows.

```sh
python scripts/analyze_arc_first.py
python scripts/analyze_existing_transitions.py
python scripts/compare_nl23_nl33w.py
python scripts/analyze_causal_pilot.py --study causal_pilot_16
python scripts/analyze_causal_pilot.py --study causal_original_shape
python scripts/audit_mechanisms.py
python scripts/trace_dynamics.py
python scripts/analyze_feedback_validation.py
python scripts/analyze_margin_effects.py
python scripts/plot_mechanisms.py
python scripts/build_final_report.py
```

Run build_final_report last: it merges study-specific results into root CSVs, while earlier individual analyzers can replace a root CSV with their own study. Individual `fresh/<study>/` files always retain their own scope. For published QA/transfer tables, scripts/audit_published.py, audit_transfer_records.py, audit_arc_nl23.py and assemble_current_evidence.py operate on hash-verified source blobs. Rebuilding those tables is optional and does not involve model inference.

The baseline raw shards contain four letter logits/probabilities for both models, question IDs and original evaluation batches. Intervention files specify each condition, original question item, target membership and all scores. Full-state donors are generated during inference and discarded after each batch; saved `.pt` files retain explicitly selected positions, not the entire long sequence. Original-shape baseline tensors include both own answer-prefix and the last sender position visible to the partner's answer-prefix. Validation tensors retain own answer-prefix for selected rows, keyed by target_rows in each `.pt`; they do not assert that this is the last partner-visible position. Load only trusted tensors with `torch.load(...,weights_only=True)`.

FP64-dot columns use the same final BF16 normalized state and exact stored frozen four output rows. They are an output-rounding diagnostic. They must not be reported as an FP32/FP64 model benchmark.

## Fresh inference in a new task directory

Preserve this evidence directory. Copy scripts/, configs/, sources/ and the pinned upstream source to a separate empty experiment directory. Do not copy old fresh/ or logs/ receipts into a new experimental run. Recreate the directory's isolated environment from configs/requirements_reproduction.txt, including the appropriate CUDA/aarch64 PyTorch wheel. The recorded environment is Torch2.11.0+cu130, Transformers5.12.0, PEFT0.21.2 on GB10. Cross-device bitwise equality is not promised.

Obtain the official repository at commit7e7fcc401a8243f94bca94070db0993d64277874 in upstream/. The bundled runtime/full source ZIPs are hash-pinned; when extracting, their upstream/ top-level directory belongs directly below the experiment directory. Download only the pinned models:

```sh
python scripts/fetch_models_manifest.py
python scripts/linux_cpu_gate.py
python scripts/run_checkpoint_pilot.py --upstream upstream --models models --out fresh/pilot_01
python scripts/run_arc_first.py --bs 16 --seconds 3600
```

Some historical wrappers include deployment-specific paths; the three commands above and linux_runtime.py are path-relative. Check the pilot's required environment/cache setup in that script and linux_runtime.configure; all caches must remain below the task directory. Fresh ARC requires sources/arc_challenge_test_pinned.parquet with SHA62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9. Full bench reconstruction can use scripts/build_bench_parquet.py and the pinned source parquet files; expected whole-bench hash is in configs/rebuilt_benchmark_manifest.json.

After the new ARC completes, reuse only selection manifests whose saved baseline agrees with the new exact shape. The prescribed run **fails** rather than silently accepting a mismatch:

```sh
python scripts/run_study_supervised.py --study causal_pilot_16 --seconds 3600 -- --study causal_pilot_16 --bs 2
python scripts/run_study_supervised.py --study causal_original_shape --seconds 3600 -- --study causal_original_shape --selection-file configs/original_shape_followup.json --bs 16 --dot64 --dissection
python scripts/run_study_supervised.py --study feedback_validation --worker-script run_feedback_validation.py --seconds 3600 -- --study feedback_validation
```

These runs are sequential. Do not launch duplicate workers. The supervisor captures actual worker exit and bounded runtime. Detach using a process launcher that closes stdin/stdout/stderr; an SSH observation timeout is not the worker outcome. No training command is required. Read the completion receipts before running analyses.

## Interpretation safeguards

Checkpoints and groups remain separate. Published nl23 has correctness only; no wrong-letter probabilities or hidden states are fabricated. Last-bridge CUT post patch and all-layer clean-sender/one-way equality are identity controls. Neighbor rows preserve original batch shape, rather than form an independent accuracy trial. Case-bootstrap bands and best-layer selection do not establish a population effect. Report the failed norm controls, BF16 ties, cross-batch instability and remaining full-FP32 limitation with the successful cases.
