# Frozen-state value calibration

This stage follows the failed 400-question training-free controller. It tests whether a very small supervised calibrator can convert input-probe responses into actual recurrent intervention value. The backbone, state actions, tokenization and scoring remain frozen. Input and confidence heads each have nine linear coefficients; a combined ablation has fifteen. Feature standardization metadata is stored separately.

Read `PROTOCOL.md` for frozen targets, data boundaries, supervision sufficiency check, comparison tests and the three-hour collection cap. `GOAL.md` preserves the full objective; a fitted head or validation AUC is not completion.

`scripts/supervise.py` runs training collection, CPU fitting, validation collection and analysis sequentially. Any phase failure stops the chain and preserves partial records. It never restarts existing results. All phase receipts and live process identities are written under `results/`; RUN_STATE alone is not evidence that a process remains live.

The scorer imports the existing corrected functions from the sibling `closed_loop_action_value_dev400_20261029` and its pinned dependencies. Keep the historical directories intact. `configs/LOCK.json` binds local sources, dependency hashes and both manifests before execution. Fitting then creates `configs/TRAIN_LOCK.json` before validation starts; analysis creates `configs/SELECTION_LOCK.json` only after applying the preregistered rule.

CPU reproduction of fitting or analysis requires NumPy, SciPy and scikit-learn and the complete stored outcome files. Use a disposable copy and retain source/dependency bytes. Fitting refuses an existing heads file or any validation outcome file; this prevents accidental refitting after evaluation. For a training-only reproduction, omit the validation outcomes and generated heads/locks in the disposable copy. GPU reproduction additionally needs the pinned official checkpoint and the audited server environment.

At initial publication this is an execution protocol and code release, not a result report. Inspect actual process receipts and completion files for current progress. No independent final benchmark is part of this collection stage.
