# CPU reproduction and integrity checks

Keep this directory beside `frozen_loop_update_value_seed2021_v1` under `experiments/`. The baseline source commit is `3fe2bac590c515056966cd7add12e62fd8951cee`. Use Python with NumPy, SciPy and scikit-learn; the recorded local environment is `D:/loopbase_cpu_20261026/Scripts/python.exe`.

Run `python -X utf8 verify_followup.py` to verify the original 167 payload hashes, conditional lock hashes, replay every saved validation prediction from saved coefficients, and independently recount paired corrections/degradations and exact McNemar p values. This does not refit or access final-task labels.

`audit_decisions.py` reproduces observable failure diagnostics. It removes the top-level gold field from SciQ/PIQA rows before diagnostics; frozen feature extraction uses no gold labels. It writes `DECISION_DIAGNOSTICS.json`.

`conditional_check.py` refuses to overwrite an existing `CONDITIONAL_RESULTS.json`. To reproduce fitting, use a disposable copy of this directory adjacent to the same sealed baseline directory and remove only the three generated `CONDITIONAL_MODELS.json`, `CONDITIONAL_PREDICTIONS.json`, and `CONDITIONAL_RESULTS.json` files from that disposable copy. Keep the script, protocol and lock bytes unchanged. Then run `python -X utf8 conditional_check.py` in that copy. Numerical floating-point details and CPU timing may vary by platform; compare discrete predictions and paired statistics. Do not use the exposed final tasks for a new policy test.

The conditional lock predates fitting and binds the protocol, script and input files. `ARTIFACT_MANIFEST.json` binds the delivered follow-up files; it excludes itself. The first-round manifest is not modified.
