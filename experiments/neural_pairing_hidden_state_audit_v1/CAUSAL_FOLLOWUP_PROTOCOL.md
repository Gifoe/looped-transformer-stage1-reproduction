# Follow-up fixed before the original-shape interventions

Checkpoint: released final nl33w only. No original or reconstructed nl23 weights. No training, Loop or new architecture.

The16-question exploratory pilot found five cases whose CUT/ON correctness partition changed when the original batch16 evaluation was changed to batch2. Same-shape repeated logits were exactly identical. Its most interesting feedback candidate is Quill on NCEOGA_2013_8_57: both CUT models and linked Ember were correct, linked Quill was wrong, and withholding reverse communication repaired Quill. This is a one-case candidate, not a discovered general mechanism.

Next replay the exact six original batches containing:

- NCEOGA_2013_8_57: feedback candidate with a correct partner output.
- Mercury_SC_400162: harm despite a CUT-correct sender; unilateral communication did not repair it in the pilot.
- Mercury_7068635: harm with a CUT-wrong sender.
- Mercury_400811: rescue although both CUT argmax answers were wrong.
- Mercury_SC_401277 and MCAS_2006_9_44: target transition changed under microbatch replay.

All original neighboring questions, row order, padding lengths and causal ticks are retained. These six batches contain95 questions. The six target cases remain distinguished from89 neighboring controls. The primary gate requires every replayed ON/CUT letter logit to match the original full ARC run exactly. No unstable case is silently replaced.

Repeat the27-condition pilot battery on these exact shapes. For the feedback candidate's original batch, additionally separate sender K/V from sender gate changes, patch each communication depth separately, remove only the reverse read at each depth, and match altered writes to factual per-token write norms. A norm-only sham must preserve original outputs. Clean-sender-all-layer equality to a direction cut is an implementation identity, not independent replication.

Compute FP64 output dot products from the SAME BF16 final normalized states and frozen output-head weights. This is a diagnostic for output rounding, not a new baseline or a full-FP32 backbone run. If the candidate's harm or repair vanishes under exact original-shape replay or this readout diagnostic, reject the corresponding robust mechanism claim.

Operational competing explanations for a given damaged receiver:

1. **Numerical/shape artifact:** the selected transition disappears under the original shape or only exists at rounded-logit ties. Resolve before semantic claims.
2. **Reverse-path mediation:** replacing the sender's states by same-question CUT states repairs the receiver. Test whether K/V content, gates, or write magnitude mediates it. This does not establish that the sender internally adopted a wrong answer.
3. **Direct harmful forward write:** damage survives clean sender/one-way execution. Test layer lesions and clean/native-state versus norm-matched random patches; do not equate sender final correctness with information at every communication layer.
4. **Useful transmission with late utilization loss:** requires evidence that meaningful sender information reaches the receiver and that a downstream intervention specifically changes its use. A successful state probe or final-state overwrite alone is insufficient.

These causal pathways can coexist across examples. The above alternatives are applied conditionally to the same failure event rather than declared globally exclusive. A layer27 CUT-state receiver patch is a positive identity control: no bridges remain after27, so CUT continuation is guaranteed. It cannot by itself localize a late utilization failure.

All effects are case-series effects, conditional on the stated selection. No benchmark-wide improvement, significance, novelty or recurrence claim is inferred from choosing the best of eight lesions. If a robust content-specific feedback mechanism survives, use additional previously untested failure cases to assess replication before proposing a research project.
