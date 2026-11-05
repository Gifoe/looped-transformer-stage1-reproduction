# Experiment B: reproducible completed input intervention pilot

See REPORT_ZH.md, hypothesis_comparison.md, numerical_validation.json and ROOT_POSTRUN_REPLAY.json. Full preregistered SHA24 matrix completed; bounded matched/B5/B6 are first4, never promoted to24. Scientific decision is STOP C. Native K1→K2 is the main behavior interval in this sample, not an identified internal causal stage/token class.

## Recover all actual records and raw vectors

The branch inherits all parent public bytes from e5764391. The pre-effect definition commit is6168ef44d74d0715059e884df826fff75e5a0a67. Frozen8source/42inputs must never be overwritten by postrun interpretations. INPUT_ARCHIVES_INDEX.json separately recovers the original gold-free inference graph/texts and labeled constructor/audit if needed.

DELIVERY_INDEX.json and RECORDS_CONTENTS.json bind a small standard tar.xz of source/CSV/log/receipts. VECTORS_PARTS_INDEX.json binds ordered fixed-size binary pieces of one standard VECTORS.npz.xz. No custom tensor transform or quantization: standard xz decode reproduces the exact148903062-byte original NPZ/1744 BF16 rawarrays. Part SHA, whole compressed SHA and raw NPZ SHA are all recorded. No model weights, tokenizer copy, private transport-coordinate wrapper or upstream private source is redistributed.

From this directory:

```powershell
python -B restore_delivery.py --verify-only
python -B restore_delivery.py
```

The restore code checks every safe regular member and refuses mismatched existing files. Verify-only still stream-decodes all vector parts and verifies the original raw hash without writing missing files. Parent ledger copies under inherited_ledgers/ are byte-identical evidence, not new cost entries and must not replace parent ledgers or be double counted.

## CPU-only independent replay, no private files or model required

Use a separate replay checkout/copy to preserve the published original receipts. Restore the standard containers and public ancestor files first, then from the parent experiment directory:

```powershell
python -B recurrent_candidate_intervention_v1/root_postrun_replay.py
python -B recurrent_candidate_intervention_v1/postrun_tables.py
python -B recurrent_candidate_intervention_v1/plot_results.py
python -B recurrent_candidate_intervention_v1/write_report.py
```

The independent Root replay uses NumPy and public source/rawpayloads, not the official analyzer or a checkpoint load; it rechecks all1744arrays,784native rows,288permutation rows,672I/J/DID rows,384B5/B6 rows,1176adjacent transitions and548 point/paired-bootstrap statistics. Its frozen result records exact agreement (including H1 actualmax0 despite a verification tolerance1e−12 declared in the independent replay checker). CPU replay does not need private RAWtransport receipts, model weights or9local model metadata files. The original official-server numeric-first analyzer and runtime file gate are separate recorded evidence; their strict source/model checks were not weakened for public replay.

For pre-effect tokenizer construction, construct_semantics.py/build_protocol.py support --tokenizer-json and the standard models path; obtain the official tokenizer.json at revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1 and verify SHA fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d. No private copy is necessary. DEPENDENCY_RECOVERY_INDEX.json describes pinned metadata recovery; locally generated snapshot_metadata.json is copied from the existing public source, not a fictitious upstream asset.

Do not rerun the already completed GPU supervisor or overwrite its outputs. A live model rerun would require the pinned9metadata plus actualweights, a new immutable output namespace, all21 actual ledgers and a separately reviewed budget. No remaining-budget spend, additional dataset or Experiment C is authorized by these files. GPU-worker cost was293.1106862188317sec; global21ledgers21182.437382869888/21600, remaining417.5626171301119. CPU archival/replay adds no GPU cost.
