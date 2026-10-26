# Reproduce the extension

The extension ZIP excludes weights, parquet snapshots, environments and cache.
Install it inside the original Stage1 package at
`extensions/baseline_suite_20261025_seed2021`. The original Stage1 artifact files
and prepared-data manifests are required; do not flatten both archives into
one directory. CPU rebuilding uses the original Stage1 artifacts/manifests and
the supplied lightweight tokenizer exports; GPU reruns also require the pinned
model/parquet snapshots. `common.py` resolves the Stage1 root two parents above
this suite.

## Frozen dependencies

- Python 3.13.9, Torch 2.11.0+cu128, Transformers 4.57.1, HF Hub 0.36.0.
- lm-eval 0.4.9, commit 452749513f817315042df9286241a61051392470.
- NumPy 2.2.6, datasets 4.4.1, pandas 2.3.3, SciPy 1.16.3,
  matplotlib 3.10.7, peft 0.17.1. Retained environment logs are authoritative.
- DoLa commit 805230e57e63ca561cb759994681b122ff6f81f0.
- Qwen3-4B-Base revision 906bfd4b4dc7f14ee4320094d8b41684abff8539.
- Ouro-1.4B revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1.
- Ouro-2.6B revision 1ed04250da1a9936042725d302e81c8fa2ab5abd.

Checkpoint loading information/configuration, source bytes/hashes, immutable
prompts and prepared request pairs are retained. Ouro requires trust_remote_code
for the pinned, reviewed source. `prepare_suite.py` documents the new 2.6B model
download and hash checks. Windows's 260-character path limit requires the short
environment `D:/loopbase_env_20261025`; the initial failed long-path install is
kept in the logs. Do not mutate the old Stage1 environment to repair this suite.

## CPU rebuild from supplied raw outputs

From this extension directory, with the original Stage1 files present:

```powershell
python -X utf8 scripts/rebuild_cpu.py
```

The entrypoint uses the calling Python interpreter, disables CUDA visibility,
forces offline tokenizer access and UTF-8, and runs the protocol tests, tokenizer
checks, all statistics/figures, frozen MC comparison, complete validation and
recovery-lineage checks, independent actual-halting audit, historical-window
audit, completion audit and report. Each child exit and log is retained under
`logs/cpu_rebuild/<UTC timestamp>/`. A missing required experiment or research
decision is an error before rebuilding; it is never silently omitted. Any failed
step stops the command and leaves `passed=false` in its receipt.

For an explicitly incomplete MC-only checkpoint, use
`python -X utf8 scripts/rebuild_cpu.py --scope mc`. Its receipt certifies only
MC rebuilding, with no math, actual stopping or full-delivery claim. This mode
has passed locally; the full command remains pending the outstanding experiments.

The full command requires the evidence-based `configs/research_decision.json`
saved after review of measured results. Rebuilding tables/figures does not call
an LLM. Bootstrap seeds are deterministic. Matplotlib PDF metadata can differ by
build time, so rebuilt PDF bytes need not equal the sealed delivered bytes.

The tokenizer exports under `sources/tokenizers/` support independent token-ID
decoding and strict/numeric EM checks without loading model parameters. Their
file hashes and checkpoint revisions are in `sources/tokenizer_export_manifest.json`.
The source model tokenizer files are copied byte-for-byte. Ouro2.6 files are also
checked against the pinned upstream Git blob IDs. Validation-prefix audits can
run while a worker is active: only complete newline-terminated records from one
fixed read are audited, and trailing bytes are explicitly counted. Such a partial
audit does not certify completion or choose parameters.

The independent actual-halting audit requires all750 real timing records and
all150 layer-count/exit/likelihood parity cases for each model. It checks exact
frozen IDs x five methods x five repeats, validation-derived settings, source
fingerprints, observed parity-layer counts, offline exit depths, every candidate
score/quality, and current-prefix decision histories. Layer hooks are excluded
from timed repeats; their complete separate parity measurements are required.
The main completion audit includes these checks. The nine auditor contract tests
use synthetic in-memory fixtures only; they neither create formal trajectories
nor establish GPU execution, latency or speedup.

`audit_reference/mc/` retains the verified 15-table MC reference, both selected
configurations, raw-file SHA-256 values and analysis-source hashes. The comparison
command requires every reference column/value to match the CPU rebuild (numeric
tolerance 1e-12), checks exact configuration equality, and rejects changed raw
records or analysis source. It covers MC only; math, actual stopping and the full
completion audit remain separate checks.

## GPU execution order

Use a fresh output directory or an exact same-code resume. A fresh GPU run copies
scripts, frozen configs, prepared data and sources, and starts with empty formal
trajectories, derived results and audit_reference directories. Prepare or retain
the validated model/environment resource receipts required by `resource_gate`
before launching MC. Its MC phase freezes its
own same-run reference after successful scoring, diagnostics and raw-record
audit; `freeze_mc_reference.py` refuses to replace a differing existing reference.
The JSONL resume
guard checks protocol/implementation fingerprints and unique record keys. It
must not be bypassed after a source change; diagnose and rerun affected outputs.

```powershell
$env:PYTHONUTF8='1'
python scripts/run_suite.py --phase mc
python scripts/run_suite.py --phase math
```

`mc` runs protocol tests, parity checks, balanced pilots, measured budget estimate,
then validation/test and CPU MC analysis. `math` runs true-cache generation parity,
five-question pilots including all validation grids and paper/depth conditions,
method-specific budget estimation, 100-question validation, validation selection,
pre-test resource sizing, aligned test generation, actual-halting parity/timing
and latency statistics. It ends awaiting research review. The `latency` phase
exists to resume that stage independently; do not rerun it concurrently.

Keep frozen prompt manifests instead of regenerating them. Formal test answers
must not be inspected to change a hyperparameter, method or sample prefix. The
resource execution count is fixed before test generation for both models and all
eight methods. Existing identical predictions are reused with an explicit marker.

On the remote Windows server, hidden WMI-launched PowerShell wrappers preserve
the supervisor across SSH disconnects. `math_launcher.ps1` waits for the actual
prior MC process exit and a successful exit record before starting. After the
observed CPU-only analysis crash, it also accepts the explicit independently
verified recovery gate proving all GPU files are complete and CPU analysis exited
successfully; the original failed supervisor exit is retained. Its hardcoded
PID is session-specific: inspect a real process handle before adapting it to a
different execution. A STATUS file alone never establishes that the GPU is free.

## Final seal and verification

After inspecting all actual results, write `configs/research_decision.json` with
`verdict`, `reasoning`, an `evidence` list and `question_answers` mapping Q1..Q12
grounded in the measured tables.
Then run `python scripts/run_suite.py --phase final`. It performs statistics,
tests, independent selection/completeness audit, report generation and packaging.

`ARTIFACT_MANIFEST.json` lists extension file SHA-256 hashes; `DELIVERY.json`
records the ZIP hash/size and original Stage1 manifest hash. The ZIP is under
`delivery/`. The packager never runs the original Stage1 packager, changes its
manifest or includes model weights. After delivery, verify ZIP and every file
hash, then rerun the CPU audit in the extracted original-Stage1 layout.

```powershell
python scripts/verify_delivery.py --zip delivery/baseline_suite_20261025_seed2021.zip
```

Keep DELIVERY.json beside the extracted manifest (it is returned separately to
avoid making an archive contain its own hash). Verify the supplied seal before
rebuilding, because regenerated figures/tables may have different file bytes.

Timings are pretokenized full MC scoring without cache, randomized within each
question with five repeats. Bootstrap units are questions, not repeats. Baseline
generation token tracing adds extra heads and is not clean latency evidence.

## Server storage layout

At the user's request, the extension-added Ouro-2.6B snapshot was relocated to
`E:\loopbase_20261025_seed2021\models\Ouro-2.6B`. Its original logical path
`D:\looped_stage1_trajectory_audit_seed2021_v1\models\Ouro-2.6B` is an NTFS
junction to that directory. All 12 copied files have identical SHA-256 hashes;
the weight hash also matches the pinned Hugging Face LFS metadata. Existing
Stage1 models, experiment inputs, protocol and inference code were unchanged.

`STORAGE_LAYOUT.md` lists the actual server folders; migration receipts are in
`logs/storage_migration_20261026.json` and `logs/storage_postcheck_20261026.json`.
Scripts `relocate_new_model.ps1` and `relocate_model_launcher.ps1` record this
one-time operation and refuse unexpected existing state. They are not required
for CPU rebuilding. A GPU reproduction only needs the pinned snapshot at the
logical model path, directly or through a verified junction; the physical E-drive
location is specific to this server. Model weights remain excluded from delivery.

## Recorded native-abort recovery

The first long Ouro1.4 validation worker stopped with native status0xC0000409
after835 complete records. Windows WER identified ucrtbase.dll; the underlying
cause was not established. Its closed raw snapshot, original status/logs/ledger
and SHA hashes are under logs/recovery_validation1p4_20261026. The entire failed
supervisor stage interval was charged conservatively to the cumulative budget.

Recovery first audited the complete prefix, then replayed three saved boundary
cases with exact token/text and baseline depth-argmax matches, and retried the
next configuration including numeric scoring and diagnostic JSONL serialization.
A Python subprocess controller captured an actual zero native exit code; the
initial PowerShell probe's missing child exit code was preserved and not accepted
as the recovery gate. Both diagnostic runs were charged. No inference code,
protocol, completed prediction or selected parameter changed.

The one-time resume uses `run_suite.py --phase math --resume-math-validation`
and requires the recorded, source/prefix-matched recovery gate. It skips completed
parities/pilots and resumes existing validation records. After1.4 reaches900,
the controller checks complete token integrity and preserved-prefix/next-case
lineage before starting2.6. The recovery commands apply to that captured state;
they intentionally refuse unexpected or already-changed inputs. Full completion
still requires both tests, actual stopping and the final audit.
