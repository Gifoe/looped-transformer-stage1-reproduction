# Result conventions

All accuracy changes use paired bootstrap (10,000 resamples, seed 2021). Intervals
are pointwise, unadjusted intervals. Sample IDs, not row position, align methods.

- `qwen_metrics.csv`: reference A0. `qwen_pairwise_metrics.csv`: explicit
  `reference` and `method` columns.
- `ouro_metrics.csv`, LoopCD transition tables and halting tables: reference is
  full-depth r4. Thus an r1 row in `ouro_metrics.csv` compares r4 against r1;
  it is not a chronological transition from r1 to r4.
- `math_paired_metrics.csv`: reference is the full-depth baseline, with actual
  greedy generation. Omega-zero identity reuses have zero additional runtime.
- In method-comparison tables, `wrong_to_correct` means reference wrong, method
  correct; `correct_to_wrong` means reference correct, method wrong.
- `trajectory_patterns.csv` uses chronological early-to-final states. Its first
  four patterns partition questions; the two oscillation patterns can overlap
  those groups and must not be summed with them. Last endpoint is actual r4.
- `math_generation_patterns.csv` compares final answers generated at different
  true depths, with different autoregressive prefixes. It is separate from the
  same-prefix MC recurrent trajectory analysis.
- Choice probabilities/log-probabilities are softmax weights over character-
  normalized candidate scores. They are not calibrated answer probabilities.
  Raw summed token likelihood and gold per-token NLL are retained separately.
- Vocabulary confidence/entropy/KL are measured at the last prompt token.
  Vocabulary and choice KL are separate variables. Hidden relative L2 and cosine
  use the final valid prompt token, comparing successive normalized depth states.
- `choice_logit_margin` is the candidate normalized-score gap. `choice_margin`
  is its softmax probability gap. Their validation threshold grids differ.
- Native gate CDF uses checkpoint weights and released BF16 arithmetic, with
  public forward readout parity. All halting results remain offline sample-level
  analyses, not measured adaptive wall-clock acceleration.
- Figure 4 lines are descriptive test Pareto envelopes of validation-frozen
  thresholds. Selected operating points use validation only. Test envelopes are
  not additional deployable parameter selections.
- `halting_matched_random.csv` intervals are the central 95% random-assignment
  distribution at exactly the same depth histogram, not bootstrap confidence
  intervals for an independently fitted algorithm.
- Mathematical generation memory fields are cumulative process-peak upper
  bounds. Qwen counters are reset for each configuration/split. Instrumented
  runtimes were measured on a shared GPU.

`completion_audit.json` verifies counts, alignment, numerical/model gates and
budget. The author Qwen patch's source gate is unresolved and explicitly excluded
from any claim of direct author-code parity. Failed runs remain in logs and the
GPU budget ledger.
