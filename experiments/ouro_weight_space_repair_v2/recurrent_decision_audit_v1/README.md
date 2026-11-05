# Reproduce the historical loop-wise decision audit

All historical parent bytes are immutable. Use branch `codex/ouro-decision-audit-20261105`; it retains `014438171c78de389af52b8ab7c822db4add43a1` and thus the requested historical `6ce4d5d4...` experiment. The execution definition is commit`809c22e168a77cf779bd271a12e37e2dbec62ff3`. See `REPORT_ZH.md` and the final archive/index for outputs.

CPU packages: Python3.10+ with numpy/scipy/tokenizers/matplotlib; torch is only needed for the synthetic head fixtures, not the historical statistical audit. Original server environment and package versions are separately recorded in `runtime_metadata.json`. Original server official all14-method exact replay is in `official_environment_exact_replay.json`; Windows old-projection FP64 differences are disclosed, not hidden behind a new tolerance.

From the parent experiment directory, after restoring the existing public historical record archives according to their INDEX/CONTENTS manifests:

```powershell
python -B scripts/r11_restore.py --verify-only
python -B recurrent_decision_audit_v1/restore_inputs.py
python -B recurrent_decision_audit_v1/audit_cpu.py --tokenizer-json PATH_TO_PINNED_TOKENIZER_JSON
python -B recurrent_decision_audit_v1/finalize_cpu.py
```

`restore_inputs.py` verifies and restores only the required inheritedR7/R10 members to this new directory; it never overwrites old files. `audit_cpu.py` accepts a tokenizer path or standard `models/Ouro-1.4B/tokenizer.json`, with the previous private audit copy only a local fallback. Obtain tokenizer metadata without model weights using `hf_hub_download('ByteDance/Ouro-1.4B','tokenizer.json',revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1')`; expectedSHA256=`fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d`. No tokenizer or weights are redistributed here.

The single-member `a4_TASK_PLAN.tar.xz` and its index recover the exact2.51MB task graph; inspect regular member paths and verify SHA before extraction. CPU planning and synthetic checks can be rerun with `prepare_a4.py TOKENIZER_JSON` and `a4_cpu_checks.py`, but the published frozen outputs remain the historical execution definition; use a separate replay workspace to avoid changing them.

Do not rerun `supervise_a4.py`: it deliberately refuses existing runtime outputs, and the only authorized model attempt has ended parity-only. The frozen500-second collector is not permission for a retry. The complete64 trajectory was not collected, and its effect-analysis outputs do not exist. The future resource proposal needs separate authorization and a new immutable output namespace. The final archive includes all actual vectors/scalars/receipts, CPU code/logs and independent Root reviews; private transport-coordinate transcripts are excluded in favor of sanitized receipts.
