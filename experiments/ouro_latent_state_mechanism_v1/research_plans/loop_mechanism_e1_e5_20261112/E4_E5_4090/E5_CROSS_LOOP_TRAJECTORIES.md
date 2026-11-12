# E5 native trajectories

Final confirmation 128 native correct counts are 69/103/110/107, with adjacent rescues/damages 40/6,9/2,1/4. K4 decreases accuracy by 3. These newly collected baselines are separate from historical2590 and current299; no monotone-gain assumption is imposed.

Historical complete ARC metadata covers Train 1119, Validation 299 and Test 1172, with 25 Train demonstrations separately identified. Source hashes and candidate scores were independently checked; the existing historical trajectories were reused rather than rerun solely for accuracy. Test native correct counts are 626/953/1000/1020; rescues/damages are 378/51, 81/34 and 33/13. Sixty-two questions first become correct at K3 and thirteen at K4; ninety-three never become correct. These facts show continuing useful and harmful later computation, without identifying its cause.

The new same-hardware full Validation baseline gives 148/243/258/261 correct, with adjacent rescues/damages 104/9, 20/5 and 7/4. These refreshed values are paired with the new interventions; they are not substituted for the historical 150/244/257/262. Eleven Validation IDs differ in at least one predicted loop.

Historical Test rescue rates conditional on being wrong before the transition decline 69.23%→36.99%→19.19%. Fewer remaining errors therefore do not arithmetically explain the entire decline in rescue count, but the remaining question sets also change in difficulty and composition. No hidden-state convergence or weakening of a single algorithm follows from accuracy alone.

Full historical per-question three-gap/tie/trajectory evidence is in inputs/historical_per_question_audit.jsonl; current-native values are in results/validation299_full_v1/*.native.json. The complete-source verification is recorded in results/ROOT_INPUT_AUDIT.json.
