$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
$target=Join-Path $root 'logs\recovery_validation2p6_20261026'
$receipt=Join-Path $target 'RECOVERY_SNAPSHOT.json'
if(Test-Path -LiteralPath $receipt){throw 'Accounting receipt already exists; do not charge twice'}
$plan=Get-Content -LiteralPath (Join-Path $root 'transport\native26_accounting_plan.json') -Raw|ConvertFrom-Json
$audit=Get-Content -LiteralPath (Join-Path $root 'results\native26_local_prefix_audit.json') -Raw|ConvertFrom-Json
$owners=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -eq 'python.exe' -and ($_.ProcessId -in @(25608,13640,27444) -or $_.CommandLine -like '*scripts/generate_math.py*' -or $_.CommandLine -like '*scripts/actual_halting.py*' -or $_.CommandLine -like '*scripts/run_suite.py*')})
if($owners.Count -ne 0){throw 'A possible experiment owner exists'}
function Assert-Hash($path,$expected){if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $expected){throw ('Hash changed: '+$path)}}
Assert-Hash (Join-Path $target 'PRESERVATION.json') $plan.preservation_sha256
Assert-Hash (Join-Path $root 'STATUS.json') $plan.expected_status_sha256
Assert-Hash (Join-Path $root 'logs\active_gpu.json') $plan.expected_active_sha256
Assert-Hash (Join-Path $root 'logs\gpu_budget.json') $plan.expected_budget_sha256
Assert-Hash (Join-Path $root 'trajectories\math_Ouro-2.6B_validation.jsonl') $plan.raw_sha256
if(-not $audit.passed -or $audit.audited_records -ne 207 -or $audit.audited_prefix_sha256 -ne $plan.raw_sha256 -or $audit.ignored_unterminated_trailing_bytes -ne 0){throw 'Local CPU prefix proof differs'}
$budgetTemp=Join-Path $root 'transport\native26_reconciled_budget.json'
$activeTemp=Join-Path $root 'transport\native26_reconciled_active.json'
Assert-Hash $budgetTemp $plan.prepared_file_sha256.'native26_reconciled_budget.json'
Assert-Hash $activeTemp $plan.prepared_file_sha256.'native26_reconciled_active.json'
$newBudget=Get-Content -LiteralPath $budgetTemp -Raw|ConvertFrom-Json
$charged=@($newBudget.runs|Where-Object{$_.pid -eq 25608 -and $_.state -eq 'native_abort_0xc0000005'})
if($charged.Count -ne 1 -or $newBudget.used_seconds -ge 180000 -or [Math]::Abs($newBudget.used_seconds-$newBudget.stage1_used_seconds-($newBudget.runs|Measure-Object seconds -Sum).Sum) -ge 0.0001){throw 'Invalid reconciled ledger'}
$budgetBackup=Join-Path $target 'ledger_before_atomic_replace.json'
$activeBackup=Join-Path $target 'active_before_atomic_replace.json'
if((Test-Path -LiteralPath $budgetBackup) -or (Test-Path -LiteralPath $activeBackup)){throw 'Atomic-replacement backups already exist; inspect partial state'}
[IO.File]::Replace($budgetTemp,(Join-Path $root 'logs\gpu_budget.json'),$budgetBackup)
[IO.File]::Replace($activeTemp,(Join-Path $root 'logs\active_gpu.json'),$activeBackup)
Assert-Hash (Join-Path $root 'logs\gpu_budget.json') $plan.prepared_file_sha256.'native26_reconciled_budget.json'
Assert-Hash (Join-Path $root 'logs\active_gpu.json') $plan.prepared_file_sha256.'native26_reconciled_active.json'
$result=[ordered]@{passed=$true;checked_utc=[DateTime]::UtcNow.ToString('o');preserved_files=$plan.preserved_files;preservation_sha256=$plan.preservation_sha256;valid_complete_records=207;raw_validation_unchanged=$true;local_cpu_prefix_audit_passed=$true;charged_failed_seconds=$plan.charged_failed_seconds;cumulative_gpu_hours=$plan.cumulative_gpu_hours;root_cause=$plan.root_cause;failed_pid=25608;next_step='CPU diagnostic and exact boundary/next-case GPU replay, captured zero exit and fresh recovery gate before resuming'}
[IO.File]::WriteAllText($receipt,($result|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
[ordered]@{passed=$true;charged_failed_seconds=$plan.charged_failed_seconds;cumulative_gpu_hours=$plan.cumulative_gpu_hours;raw_records=207;raw_sha256=$plan.raw_sha256}|ConvertTo-Json
