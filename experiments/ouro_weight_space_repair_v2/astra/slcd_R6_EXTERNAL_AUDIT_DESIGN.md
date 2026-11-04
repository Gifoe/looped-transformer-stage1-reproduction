# Independent final external acceptance

Implementation: [slcd_r6_external_audit.py](../scripts/slcd_r6_external_audit.py). This is an additive CPU-only audit, not a change to inference, ranking, selection, locks or active GPU jobs.

From the experiment root on the server, after both tasks and their chain complete:

```sh
env/bin/python -B scripts/slcd_r6_external_audit.py
```

Output: `results/slcd_R6_EXTERNAL_ACCEPTANCE.json` with separate HellaSwag and OpenBookQA results, verified artifact/source hashes, coverage counts and explicit limitations. A successful run prints only the accepted final counts and output path. Missing or inconsistent prerequisites cause a nonzero exit; the script does not wait, score a model, or accept incomplete datasets.

Optional task-specific rechecks:

```sh
env/bin/python -B scripts/slcd_r6_external_audit.py --task hellaswag
env/bin/python -B scripts/slcd_r6_external_audit.py --task openbookqa
```

These write task-specific acceptance receipts. **Both task COMPLETE files and the chain COMPLETE remain required even in task mode**, preventing any accidental audit of the still-running experiment. Combined mode also requires that every new external prior cache entry is referenced by at least one of the two completed tasks.

The independent audit verifies:

- Final/Dev/source/controller/preparation/config hashes, the accepted ARC threshold gate, exact original checkpoint file hash and tokenizer asset hashes, and locked package versions.
- Official raw dataset source receipts/parquet hashes and installed harness task YAML/utils hashes. The manifest's complete ordered IDs, input bytes, prompt SHA, choice order and character denominators are bound to the locked manifest; the audit does not rerender it through ConfigurableTask.
- All eight original and all eight prior readouts: raw LL, one character normalization, finite values, first-argmax prediction and correctness flags.
- All five final rankings. Native/Fixed/Adaptive/A1 are independently reconstructed; the R6 projection is independently derived from the candidate arrays and cross-checked using **only** frozen `selected_score`. `all_scores` and worker validation helpers are never imported or called.
- Old-main rows against their immutable source/protocol/input authorities, plus the three-row replay receipt where applicable. The CPU audit verifies the replay receipt and coverage; it does not recreate a GPU replay.
- Old ARC prior cache authority, each old/new cache key and entry digest, raw LL references, candidate continuation bytes and actual prefix/target IDs. Each context is independently encoded with the source-hash-verified harness `_encode_pair`; unequal contextual/prior targets are explicitly allowed and checked, never truncated or silently equated.
- Both comparative summary CSVs, the complete paired-transition CSV and every full-candidate score CSV row. Duplicate/missing IDs, methods and choices fail. All R/D counts, exact McNemar tests and seeded 50,000-replicate paired bootstrap intervals are recomputed independently.

Prerequisites include all external manifests, raw data/parquet and source receipts, full score JSONLs, four CSVs per task, both COMPLETE/replay receipts, chain COMPLETE, frozen locks and source files, original model/tokenizer files, old HH partial-main authority, ARC prior COMPLETE/score/cache files, and the new external prior cache. Use the server's original environment: package versions are checked exactly. No packages are installed, no network access is made, and no model, `torch`, worker module or `torch.load` is imported by the audit. Checkpoint validation is a streaming byte hash only. The official small pair-encoding function is extracted from the installed harness source, hash-checked, and executed with a tokenizer-only backend.

Memory is bounded by manifest/compact output indices and old/new prior caches; the large candidate/transition CSVs are streamed. JSONL readouts are processed one question at a time. The script rehashes verified files before writing acceptance to detect concurrent mutation.

Synthetic verification command (does not require model files, locks or results):

```sh
python -B scripts/slcd_r6_external_audit.py --self-test
```

AST parsing and synthetic checks passed during implementation. Tests cover independent ranking, corrupted normalization rejection, paired statistics, typed CSV checks, and altered cache keys/target-boundary rejection. No real or partial external outcomes were read during development. No claim is made that these synthetic checks constitute the eventual full acceptance run.

Acceptance does not establish independent regeneration of the likelihoods, a new causal mechanism, runtime speedup, multiplicity-adjusted significance, publication timing, or general superiority. It authenticates and independently recombines the completed artifacts under the frozen method.
