# Execution issues and deviations

- Author Training-Free Loop repository: repeated public/API/connector 404.
  Fixed numerical A0-A4 recipe is evaluated with an independent implementation.
  Direct parity against the author patch cannot be established. Source audit is
  in `sources/IMPLEMENTATION_AUDIT.md`.
- Base Anaconda NumPy/MKL and Torch triggered duplicate OpenMP initialization.
  An isolated D-drive venv with NumPy's pinned pip wheel resolved the import;
  `KMP_DUPLICATE_LIB_OK` was not used. The base environment was left unchanged.
- First background preparation launched via an SSH child was terminated with
  the SSH job. A hidden WMI-launched supervisor persists correctly. Initial
  logs were retained with `.attempt1` suffixes.
- Full streaming transfers stalled. Verified HTTP ranges and a PyPI mirror
  were added. Larger SFTP downloads stalled even without prefetch. Final
  artifact retrieval uses compressed stdout over an authenticated SSH channel.
  A single large ZIP stdout transfer also stalled; the transport now bounds
  each chunk to 256 KiB and checks decoded lengths before final archive/hash
  verification. SFTP remains in use for small source uploads.
- Initial independent Qwen K=1 Euler identity failed at relative logits L2
  0.01737 (maximum absolute error 0.3125), while naive K=1 was exact. This was
  isolated to BF16 cancellation in x+(y-x). K=1 directly returns y after repair.
  No fixed K=2 or K=6 update was changed. All sixteen repaired identity cases
  had zero logits error, and canonical cache tests passed. Failed runs remain
  in diagnostic logs and GPU budget records.
- Choice results use a fixed zero-shot supplementary protocol. They must not be
  compared numerically to the LoopCD paper's 25-shot ARC headline table. No
  author-reported score is substituted for a local run.
- Qwen evaluation was interrupted by a Windows sharing violation when an atomic
  progress-file replacement overlapped with a status read. Raw JSONL had already
  been flushed. Atomic writes now retry transient sharing failures; progress is
  updated every ten questions. Resume deduplicates immutable sample IDs. This
  interruption does not change model computation or rerun completed predictions.
  Qwen full-vocabulary KL versus A0 was an optional in-memory diagnostic and is
  absent after restart. Complete choice-level KL versus A0 is reconstructed for
  every configuration; sparse full-vocabulary KL is not used in summary claims.
- Halting is offline, sample-level, and reports average recurrent depth and
  theoretical recurrent work only. It does not demonstrate token-level exit or
  wall-clock acceleration.
- Statistical QA corrected the two oscillation patterns to require their last
  element at the actual final depth r4. An earlier draft counted any triple
  inside four rounds. Raw predictions were unchanged; the analysis and figures
  were rebuilt and a counterexample test now fixes the endpoint semantics.
- Margin definitions are kept explicit: the primary paper-style score/logit gap
  is derived from saved candidate scores; the earlier probability-gap diagnostic
  is retained as a separately named curve. Both threshold grids use validation
  only. Raw logits/scores and predictions were not modified.
- The requested protocol's author-source gate remains unresolved. The frozen
  requested configuration is retained verbatim; `configs/execution_scope.json`
  explicitly records the independent supplementary scope. Final direct web and
  GitHub API checks both returned 404; evidence is retained under `sources/`.
- Final report QA clarified the native-gate grid's budget feasibility: its
  smallest positive threshold needs 3.07 validation rounds, so the <=3 operating
  point falls back to fixed depth one. The predeclared <=3.5 point is separately
  reported and is competitive near three rounds. No grid, threshold-selection
  rule or test prediction was changed. Additional mathematical absolute-cut
  sensitivity is explicitly descriptive and does not select deployment settings.
