# Follow-up target: at least733/1172 (62.5%)

Current verified ARC test remains **726/1172**. The target has **not** been reached. No follow-up method has been evaluated on ARC test yet; no new result is implied by code being present.

The original A1 method/lock and all older V2 files are preserved. The user's higher target initiated further training-only research in this same folder. All1093 train questions and official validation had already been used, so deterministic5-fold slices are descriptive robustness checks, not independent validation.

| Follow-up family | Best train correct/1093 | Incumbent | Decision |
|---|---:|---:|---|
| R2 Fisher late-residual decorrelation, Euclidean/matched-norm/cosine controls |689|690|Rejected; no test|
| R3 four candidate-score compositions |691|690|Only+1; not tested|
| R4 Fisher energy/exact-KL/uniform energy budgets and Adaptive+late control |687|690|Rejected; no test|
| R5 fixed answer-prior corrections |not run|690|Code statically audited and algebra checked; awaits server connection|

R2/R3 have complete development summaries and fold tables here. R4 completed on server with zero baseline errors and a locked recipe; this local branch currently contains its SSH-observed aggregate snapshot, not the untransferred full server records. R5's six formulas are fixed in source; no R5 development or test lock has been created and no model job started.

R4/R5's predeclared test guard requires at least+7 correct on train1093 vsA1 and at least3/5 nonnegative folds, before one primary can be frozen. This does not guarantee seven additional test corrections. No coefficients are chosen from new test results.

Full HellaSwag for original A1 was intentionally yielded after1956 saved rows to prioritize the user's higher accuracy target. Those rows and all worker time remain preserved; the partial sample is not reported as full benchmark accuracy. Initial network download failures were also preserved and charged. The independent additional6h GPU-worker ledger remains in force.

The current operational blocker is SSH connectivity to the designated GB10:10.20.11.80:22 timed out after the earlier session disconnected. The user reports the server is online. Local TCP/ping fail and no Tailscale route to this private address is present. Restore the route/VPN or provide another reachable address to resume. R5 source is prepared, not executed.

Research claims are limited: projection guidance, Fisher logit geometry, KL trust regions and PMI calibration have prior art. See astra/slcd_FOLLOWUP_RELATED_WORK.md. No novelty or causal mechanism is inferred from these negative training comparisons.
