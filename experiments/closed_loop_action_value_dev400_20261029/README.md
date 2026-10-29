# Closed-loop action value: 400-question development intervention study

This directory extends the frozen eight-question prototype to all 400 exposed OpenBookQA discovery questions. It tests whether the same input-probe variable selects beneficial recurrent-state interventions, rather than only changing final readout scores. No independent confirmation claim is made.

The mechanism and analysis were frozen in `PROTOCOL.md` and `configs/LOCK.json` before running the study. A numerical position-packing correction is documented separately in `PACKING_AMENDMENT.md` and `configs/PACKING_AMENDMENT.json`; the first attempt and its 52 complete records remain preserved. The controller, intervention strengths and statistical comparisons are unchanged. The corrected full dataset is `results/development_packed.jsonl`.

When complete, `FINDINGS_ZH.md` reports actual outcomes, including fixed-action, confidence, terminal-only, shuffled-action and oracle controls. `results/VERIFICATION.json` records independent decision replay and preservation checks. Scientific plots come from the measured analysis JSON.

Keep this directory beside the original `frozen_loop_update_value_seed2021_v1`, `closed_loop_action_value_20261029` and `update_value_followup_audit_20261029` directories. In the existing CPU environment, run:

```text
python -X utf8 scripts/analyze_packed.py
python -X utf8 scripts/verify_results.py
python -X utf8 scripts/plot_results.py
python -X utf8 scripts/render_findings.py
```

The rendering script also requires the supplied process-based `results/BUDGET.json`. GPU reproduction uses the previous experiment's pinned official checkpoint and environment; use a disposable copy with empty generated result files, and launch `scripts/supervise_packed.py` so the process is bounded and logged. This is a full 400-question recomputation. Do not modify locked source bytes or treat previously exposed final tasks as new validation data. The first attempt is retained only as operational evidence and is not mixed into the corrected analysis.

Model parameters stay frozen; no head training, new threshold fitting, true early exit, or serving-efficiency claim is part of this study.
