# V2 delivery engineering review

Code-only review of package_v2.py, unpack_artifacts.py, audit_v2.py, verify_selected_checkpoint.py and assemble_report.py. Also checked the supervisor/finisher interfaces. No launch-blocking defect was found. No numerical, selection, pipeline, or V1 code was modified. No held-out outcome files were read and no GPU job was run.

## Verified interfaces

- Packaging includes all 14 exact candidate bundles and the selected copy. The original model is excluded. Weight tar members retain experiment-relative paths, are compressed once, split into ordered 4 MiB parts, and checked against the source bytes before indexing. The restore utility verifies part hashes and the assembled archive hash, refuses archive links and traversal, confines writes to the restored V2 root, and preserves differing existing files.
- Selected/candidate manifests are both directly published and archived with identical bytes. Statistics/factors are directly indexed; large per-question raw-state tensors are explicitly remote-only. ARTIFACT_INDEX excludes its own self-hash; PUBLIC_FILE_INDEX then includes ARTIFACT_INDEX. This avoids a circular hash dependency.
- Records archive content intentionally precedes PACKAGE_COMPLETE; active package/finisher logs are excluded. The result claim is therefore a stable snapshot, not a live log tail. The package is fail-closed rather than automatically resumable: an existing zip/temporary weight archive/part is rejected. If interrupted, preserve and inspect that attempt before retrying into a clean output location. This does not affect a first successful run.
- The selected-checkpoint gate reads a frozen training question and compares native K1–K6 plus both K4 LoopCD scores against that same selected candidate's screen record. Method names match the pipeline. Shared-parameter observations expect six invocations per candidate continuation. All named parameters are hashed before editing, protected during the edit, and checked after exact restoration. A second bundle load replays the saved BF16 bits, not a fresh factor GEMM. These are reproducibility checks, not accuracy-based selection.
- Audit row counts agree with the analysis: 22 search rows, 15 edit rows, six wide comparison rows, 12 long rows, 7032 paired rows, six paired statistics and 12 generalization rows. Budget receipt names and fields match supervise.py. V1 preservation checks compare the original tracked size/mtime inventory and the model hash; report wording correctly describes that scope rather than claiming a content hash of every V1 artifact.
- Report lookups match the current analysis/config/gate schemas. Unsupported LoopCD depths remain NA, and the final selected edit is used consistently for plain and combined comparisons. No result-dependent editing or new method choice is introduced by these delivery helpers.

## CPU verification

Synthetic archive round trip passed with 13 deliberately small ordered parts. It checked exact restored record/weight bytes, repeat extraction of identical files, preservation of a conflicting existing file, and rejection of parent-directory traversal. Evidence: `astra/DELIVERY_SYNTHETIC_CHECK.json`. This used synthetic bytes in a fresh V2 tmp directory, not real model/result files. All five scripts parsed successfully using UTF-8. Real archive compression duration/size and the final GPU gate remain execution checks for the finisher.

Reviewed SHA256:

| Script | SHA256 |
|---|---|
| package_v2.py | a77be8fe5b14442c81b267edd1fa523861ee7c9dd7f7794c875ab63e64c5b322 |
| unpack_artifacts.py | bb4b43f11740fa283bddd97fe6e771b6283b09116a8bcbb5745135ad940ac008 |
| audit_v2.py | 8a02c659ea88777d012d00c1454336a95d440f2a9a9b8b31e59b9bb2b746bccf |
| verify_selected_checkpoint.py | ac273710e7abed776400f13707f6d832e464919524690be829a80056af07e84e |
| assemble_report.py | 663e0e97bd5c7604c12bf42f6a86434e0ef78f0b9aac4ca8070f976fb56c8bfe |
