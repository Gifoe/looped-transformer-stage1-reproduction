$ErrorActionPreference='Stop'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
& powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ($root+'\scripts\relocate_new_model.ps1') *> ($root+'\logs\storage_migration_wmi.log')
$code=$LASTEXITCODE
[ordered]@{exit_code=$code;finished_utc=[DateTime]::UtcNow.ToString('o');pid=$PID}|ConvertTo-Json|Set-Content -LiteralPath ($root+'\logs\storage_migration_exit.json') -Encoding UTF8
exit $code
