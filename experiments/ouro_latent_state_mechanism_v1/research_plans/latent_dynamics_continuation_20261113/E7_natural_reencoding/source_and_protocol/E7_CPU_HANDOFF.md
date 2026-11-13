# E7 CPU analysis execution handoff

CPU implementation/analysis owner: current Sol. Astra's formal order predates new variant outcomes. CPU source was frozen after technical parity outputs existed, before any new variant scientific outcome/label comparison was read. Do not relabel that chronology.

- GPU collector: existing actual PID16162, managed only by root. Never restart or launch another GPU job from this CPU worker.
- CPU queue: actual PID16978, command `/root/autodl-tmp/ouro-e6a/venv/bin/python -u /root/autodl-tmp/ouro-e7/reencoding/E7_CPU_QUEUE.py --root /root/autodl-tmp/ouro-e7/reencoding --collector-pid 16162 --poll-seconds 30`.
- Queue log: `/root/autodl-tmp/ouro-e7/reencoding/cpu_queue.log`.
- Atomic cursor/status: `/root/autodl-tmp/ouro-e7/reencoding/analysis/CPU_QUEUE_STATE.json`.
- Scientific analysis: `/root/autodl-tmp/ouro-e7/reencoding/E7_REENCODING_ANALYSIS.py`, SHA `ee261928369875a03748895062ec3961ce9753690a22ddc3cbeb3f959cff00eb`.
- Queue source SHA: `26760a50ad75ef5df3a2a1c0212cf1060bbe76b4f8feb0d9ec4f8dbe7026318f`.
- Source/permission freeze: `analysis/E7_ANALYSIS_SOURCE_FREEZE.json`, SHA `a56ce38c2ec1e927fc73f98f28132923b1d59832841feb9546f17628eef8a1d7`.
- Pure hand-constructed mathematical QA: `E7_CPU_SYNTHETIC_QA.py` and `.json`; passed pure-P/pure-S/interaction grids, 3/5-choice rectangles, deterministic10000 block resampling, fixed-pair FP64 common-offset removal and semantic-index exact ties. It opened no new outcome or label files.

The queue deeply verifies only Train unit JSON before Train freeze: canonical entire-variant input hash, payload checksum, source/model identity, IDs/p/s, first-layer/body/head counts, native coverage and identity bridge. It does not parse new Validation outcome JSON before Train analysis/frozen decisions. It observes collector `/proc` start tick and command, not only PID existence. Observation intervals never trigger a restart.

Train requires17504 complete units from1094 questions (fourchoice1092 is causal/method main; unique three/five extensions separate). Source analysis then writes `E7_TRAIN_FROZEN_DECISIONS.json`, selecting gate only on original identity allTrain1094, and deciding fixed-pair profile eligibility only on main fourchoice Train. Only then does the queue read4776 Val units from299 questions (fourchoice295). It additionally requires global collector `COMPLETE.json` with22280 units before independent once-only Validation evaluation and verifies its per-file SHA records against all verified units.

If collector actually terminates without full atomic completion, queue writes `CPU_QUEUE_TERMINAL_FAILURE.json`, logs the exception and exits. It never turns partial coverage into a full-cohort conclusion and never restarts GPU. Successful end markers are `TRAIN_COMPLETE.json`, `VALIDATION_COMPLETE.json`, `CPU_QUEUE_COMPLETE.json`.

All planned causal contrasts, two tie policies, alloffset transition/risk tables, question gaps, consistency/oracle distinctions, diagonal boundary, unique fixed pair, original E6 controls, gate fixed thresholds and frontier/tie selection are fixed in source. Primary comparisons are interpreted only with `stratum == '4_choice'`; extension contrast forms remain descriptive and separate. All GPU budget/timing claims remain unmeasured replay unless root subsequently executes the formal online eligibility branch. No fitted heads, new threshold grid, Test inputs, hidden-state intervention, or Git publishing occur in this worker.
