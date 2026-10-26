$ErrorActionPreference='Stop'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
$owner=Join-Path $root 'logs\native26_validation_resume_owner.json'
if(Test-Path -LiteralPath $owner){throw 'Resume launcher has already run; inspect actual processes before any new launch'}
$gate=Get-Content -LiteralPath (Join-Path $root 'results\native_abort_validation2p6_gate.json') -Raw|ConvertFrom-Json
if(-not $gate.passed -or $gate.recovery_model -ne 'Ouro-2.6B'){throw 'Missing second-model recovery gate'}
$source=Join-Path $root 'scripts\run_suite.py'
$copy=Join-Path $root 'logs\recovery_validation2p6_20261026\resume2p6_controller_loaded_source.py'
if(Test-Path -LiteralPath $copy){throw 'Loaded-source snapshot already exists'}
Copy-Item -LiteralPath $source -Destination $copy
[ordered]@{launcher_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');controller_source_sha256=(Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash.ToLower()}|ConvertTo-Json|Set-Content -LiteralPath $owner -Encoding UTF8
& 'E:\Anaconda\python.exe' -u -X utf8 -X faulthandler $source --phase math --resume-math-validation --validation-recovery-gate results/native_abort_validation2p6_gate.json *> (Join-Path $root 'logs\native26_validation_resume_stdout.log')
$code=$LASTEXITCODE
[ordered]@{exit_code=$code;finished_utc=[DateTime]::UtcNow.ToString('o');launcher_pid=$PID}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $root 'logs\native26_validation_resume_exit.json') -Encoding UTF8
exit $code
