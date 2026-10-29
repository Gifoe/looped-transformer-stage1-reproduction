# Closed-loop action value: mechanism prototype

This follow-up implements the user's clarified priority: an intermediate variable should drive continuation or feedback into subsequent recurrent computation. It preserves the first-round No-Go result and does not claim a successful controller or improvement over LoopCD.

See [DESIGN_ZH.md](DESIGN_ZH.md) for the candidate variable, limitations, action definitions and evidence requirements; [PILOT_FINDINGS_ZH.md](PILOT_FINDINGS_ZH.md) reports the actual eight-question implementation check. The immutable [pilot lock](configs/PILOT_LOCK.json) predates execution.

Run `python -X utf8 scripts/verify_pilot.py` in a checkout where the previous experiment is available as the sibling directory `frozen_loop_update_value_seed2021_v1`. This replays decisions from saved input-probe losses and verifies source/data hashes, baseline parity evidence, propagated interventions, and original artifact preservation. It does not rerun the model.

GPU reproduction requires the same pinned Ouro checkpoint and environment as the previous experiment. `scripts/pilot.py` imports the previous scorer's tokenization, causal probe indexing and LoopCD logit function read-only. Run from a disposable directory copy with empty `results/`, preserving the sibling directory layout; `scripts/supervise_pilot.py` bounds the unchanged worker at 1,200 seconds and preserves its process receipt. It refuses an existing pilot JSONL. The historical experiment is never rewritten.

The first direct PowerShell worker exited without a receipt or result rows. Its native cause is unknown; empty log and the operational recovery record are preserved. A subprocess supervisor recovered execution without changing the locked worker. Its complete wall time plus a conservative bound for the initial attempt are recorded in the budget receipt. This pilot is complete; no full development or independent confirmation job is running.
