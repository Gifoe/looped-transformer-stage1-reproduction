# Reproduce the frozen calibration stage

The initial public release contains the frozen execution protocol and an execution-start receipt. Completion requires all phase exits, both complete outcome files, fitted coefficients, TRAIN_LOCK, the validation analysis and SELECTION_LOCK. A start receipt alone does not establish completion.

Keep the sibling historical experiment directories intact. Their exact source/data dependencies are listed in `configs/LOCK.json`; preserve original file bytes. The GPU collector imports the corrected state-intervention implementation from `closed_loop_action_value_dev400_20261029`, which uses the pinned official Ouro-1.4B checkpoint and original scorer dependencies. It uses the server's audited Python environment and model revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

For a new GPU reproduction, use a disposable copy of this experiment and its dependencies. Remove only the generated outcomes, generated heads and generated locks in that disposable copy before launch. Retain `configs/LOCK.json`, its hash-bound files, and all historical dependencies. The supervisor intentionally refuses an existing RUN_STATE; the collector refuses existing output files. Do not delete production or historical artifacts to make a rerun start.

The frozen chain is:

```powershell
python -X utf8 -u scripts/supervise.py
```

It collects 3,200 training questions, fits the fixed heads on CPU, writes TRAIN_LOCK, collects 800 validation questions and applies the frozen analysis. The hard chain cap is three hours. No automatic retry is permitted. Each child has an exit receipt; failures and partial outcomes remain charged. Fewer than 20 beneficial or harmful events for any action stops the chain before fitting.

For CPU reconstruction after the complete artifacts are available, use Python with NumPy, SciPy, scikit-learn and Matplotlib. In an unchanged completed copy, run:

```powershell
python -X utf8 scripts/verify_completed.py
python -X utf8 scripts/report_completed.py
```

The verification independently reconstructs ridge coefficients with a closed-form solve, validates all stored decisions, source/dependency hashes, candidate scoring, recurrent state propagation, head-freeze timing and the selection rule. The report renders the stored results and secondary acceptance accounting; it does not fit or choose a new policy. PNG and PDF figures require visual inspection before final delivery.

To reproduce fitting itself, make a separate training-only copy containing the frozen sources, complete training outcomes and dependencies, but omit generated heads, TRAIN_LOCK, validation outcomes and subsequent outputs. Run `scripts/fit_heads.py` there. Never rerun fitting in the completed evidence directory. Validation is historically exposed development data; no independent confirmation is implied by either reproduction command.
