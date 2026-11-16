# Ouro K1/K2 Temporal Causal Bridge

Frozen-checkpoint, two-loop causal study in a controlled two-hop relation task. Sol implemented, executed and self-audited all work. No independent agent participated. Start: b9f782d07cfc63ace6787169b98776e4bcbbb127. Prior single-edge negative results remain unchanged.

Read FINAL_MECHANISTIC_CONCLUSION.md first; detailed A/B/C reports and all phase JSON summaries retain successful and unsuccessful contrasts.32 development +64 confirmation groups,768 directed units /768 distinct text worlds. Clustered intervals, exact ties, source hashes and frozen choices included. The principal confirmed map→later sensitivity occurs in fixed L9–16; L1–8 is weaker; L17–24 is inconclusive. This is a mechanism under the stated synthetic family/intervention, not a universal ARC explanation or unique symbolic algorithm.

## CPU replay

Python3.10+, NumPy1.26.4 (analysis reference). No torch, checkpoint, GPU or network required.

```bash
python code/reproduce_release.py
```

The script verifies PUBLICATION_MANIFEST bytes, assembles lossless gzip shards, decompresses the required ALL_QUESTION_RESULTS.jsonl/INPUTS/UNITS/reference-provenance artifacts into a temporary folder and reproduces all six A/B/C phase summaries, reciprocal identities, scalar/input audit and dose audit exactly. Large JSONL files are gzip-compressed losslessly in Git; no samples or fields dropped. Statistical replay is distinct from independently rerunning the neural interventions.

## Neural reproduction

Requires the exact four checkpoint/config/source files in SOURCE_RUNTIME.json; weight files are not redistributed. Reference software: PyTorch2.5.1+cu124, Transformers4.55.0, BF16 original SDPA, one4090. Use an empty work directory and the same --work-dir for all stages, in this order:

```bash
python code/reproduce_gpu.py --model-dir /path/to/model --work-dir /path/to/fresh_run --stage technical-A
python code/reproduce_gpu.py --model-dir /path/to/model --work-dir /path/to/fresh_run --stage A-development
python code/reproduce_gpu.py --model-dir /path/to/model --work-dir /path/to/fresh_run --stage A-confirmation
python code/reproduce_gpu.py --model-dir /path/to/model --work-dir /path/to/fresh_run --stage B-development
python code/reproduce_gpu.py --model-dir /path/to/model --work-dir /path/to/fresh_run --stage B-confirmation
python code/reproduce_gpu.py --model-dir /path/to/model --work-dir /path/to/fresh_run --stage technical-C
python code/reproduce_gpu.py --model-dir /path/to/model --work-dir /path/to/fresh_run --stage C-development
python code/reproduce_gpu.py --model-dir /path/to/model --work-dir /path/to/fresh_run --stage C-confirmation
```

The adapter replaces only project/raw/model path strings and creates one fresh180min deadline; original source versions are preserved. Model loader verifies checkpoint and actually imported source hashes. Frozen original gates are copied as documented receipts; each new B/C run additionally verifies fresh predecessor native/condition full-vector hashes. These scripts preserve original fail-fast numerical checks. Hardware/software changes may fail bitwise identity; do not relax gates to manufacture success.

## Artifact scope

ALL_QUESTION_RESULTS: one row/directed unit; top-levelA, nested region_B and path_C. Every candidate logit, full-vocabulary-normalized log probability, tie, reference identity and realized per-row dose retained. REFERENCE_KV_PROVENANCE maps actual W0 sources to deduplicated byte-identical prefix hashes at all48 sites. G references use each condition's actual same-layer-input Z/T tensors; technical witnesses and all complete49152 outputs remain server-side with RAW_TENSOR_MANIFEST hashes (about3GB), not redistributed. The public CPU replay validates scalars/statistics, not unavailable raw KV. Model and tokenizer are not in release. No credential/access files, original SSH logs, private host configuration, checkpoint weights or unrequested prior raw datasets are included.

Original neural sources are byte-preserved; code/source_versions/prepare_B_original.py supplies exact legacy generator provenance. prepare.py retains original absolute paths; use frozen INPUTS for neural reproduction. Public report builders are archival utilities; reproduce_release.py is the supported portable scalar entrypoint.

The randomized control is approximate BF16 SDPA-output norm matching across all24 H layers. Partial-window interventions are not each separately dose matched. Reciprocal A symmetry and construction checks must not be interpreted as independent mechanisms. See PATH_BLOCK_AND_RESTORE.md and RANDOM_DOSE_AUDIT.json for realized errors.
