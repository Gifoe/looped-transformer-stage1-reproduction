# Server storage layout

Updated UTC: 2026-10-25T16:44:58.4322384Z

| Purpose | Actual server path |
| --- | --- |
| Sealed Stage1 and existing Qwen/Ouro1.4 snapshots | D:\looped_stage1_trajectory_audit_seed2021_v1 |
| Extension scripts, configs, raw results, logs and reports | D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021 |
| Extension Python environment | D:\loopbase_env_20261025 |
| Newly added Ouro-2.6B snapshot, physical location | E:\loopbase_20261025_seed2021\models\Ouro-2.6B |
| Compatible original model path, NTFS junction to E | D:\looped_stage1_trajectory_audit_seed2021_v1\models\Ouro-2.6B |
| F drive | Inspected, approximately 218 GiB free at preflight; no project files stored there |

All 12 snapshot files were copied with identical SHA256 hashes.
Migration evidence: logs/storage_migration_20261026.json.
Existing Stage1 files, model bytes, protocol and inference code were unchanged.
The junction preserves model loading paths; it does not allocate a second copy.
No currently active Ouro-2.6B worker existed during the migration.

## Read-only resource inventory

Checked UTC2026-10-25T19:32:02.746872Z while the2.6B validation worker was active.

| Task-owned folder | Apparent file bytes |
| --- | ---: |
| D:\looped_stage1_trajectory_audit_seed2021_v1, including caches/failed env/transport snapshots; E junction pruned | 11,373,592,649 |
| D:\loopbase_env_20261025 | 36,065,079 |
| E:\loopbase_20261025_seed2021 | 5,340,864,043 |
| Total, counting each physical model snapshot once | 16,750,521,771 |

The total is16.75GB (15.60GiB), below the preferred60GB target. The existing
shared E:\Anaconda runtime is not a task-owned allocation. This is an
apparent-file-size inventory, not an NTFS allocated-cluster measurement.
No unrelated folders were traversed and no files were moved or deleted.

Available physical RAM was54,647,070,720bytes (50.89GiB), so RAM was not exhausted
at this checkpoint. Free disk bytes were D12,139,634,688, E144,741,298,176,
F233,723,994,112. The D free-space reserve remains5GiB. Detailed folder/file
counts and skipped-junction paths are in logs/storage_inventory_20261026.json.

Runtime diagnostics additionally created two small D-drive sibling folders:
`D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021_cold_default1`
and `...\baseline_suite_20261025_seed2021_cold_debug1`. They hold byte-identical
producer/input copies and separate diagnostic trajectories. Their `logs`
junctions point to the extension's single authoritative GPU ledger. No model
weights were duplicated. Diagnostic row208 is excluded from the formal207-row
validation prefix. Full diagnostic receipts are in the extension's
`logs/native26_cold_context/`.

The completed affinity diagnostic additionally used
`D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021_affinity_ecore1`.
Its24-question diagnostic inputs and216-row output are isolated from the formal
100-question validation. Its `logs` junction shares the same authoritative
ledger. No weights were duplicated. The compact evidence archive and explicit
new-nine-row file are retained under `logs/native26_affinity_context/`.

Independent CPU checking created a local-only support folder,
`D:\chenyu-iclr\experiments\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021_affinity_ecore1_cpu_audit`.
It contains copied audit inputs/tokenizers, no model parameters, and is excluded
from the GitHub checkpoint because the diagnostic evidence is retained in the
main extension. No new F-drive task folder was created.
