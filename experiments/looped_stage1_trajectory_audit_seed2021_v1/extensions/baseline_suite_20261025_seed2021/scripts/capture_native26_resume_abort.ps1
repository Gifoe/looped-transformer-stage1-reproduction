$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
$target=Join-Path $root 'logs\recovery_validation2p6_resume1_20261026'
$status=Get-Content -LiteralPath (Join-Path $root 'STATUS.json') -Raw|ConvertFrom-Json
$active=Get-Content -LiteralPath (Join-Path $root 'logs\active_gpu.json') -Raw|ConvertFrom-Json
if($status.stage -ne 'failed' -or $status.error -notlike '*3221225477*' -or $active.pid -ne 35364){throw 'Unexpected native-abort identity'}
$owners=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -eq 'python.exe' -and ($_.ProcessId -in @(35364,44328) -or $_.CommandLine -like '*scripts/generate_math.py*' -or $_.CommandLine -like '*scripts/actual_halting.py*' -or $_.CommandLine -like '*scripts/run_suite.py*')})
if($owners.Count -gt 0){throw 'A possible experiment owner still exists; do not preserve/relaunch yet'}
[void](New-Item -ItemType Directory -Path $target -Force)
$names=@('STATUS.json','logs/active_gpu.json','logs/gpu_budget.json','logs/generation_validation_Ouro-2.6B.log','logs/native26_validation_resume_stdout.log','logs/native26_validation_resume_exit.json','results/math_prefix_audit_Ouro-2.6B.json','trajectories/math_Ouro-2.6B_validation.jsonl','results/model_load_Ouro-2.6B.json','logs/recovery_validation2p6_20261026/resume2p6_controller_loaded_source.py')
$manifest=[ordered]@{}
$missingOptional=@()
foreach($name in $names){
    $source=Join-Path $root $name
    if(-not (Test-Path -LiteralPath $source)){
        if($name -eq 'logs/native26_validation_resume_exit.json'){$missingOptional+=$name;continue}
        throw ('Required preservation source missing: '+$name)
    }
    $destination=Join-Path $target ($name.Replace('/','__'))
    $digest=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower()
    if(Test-Path -LiteralPath $destination){
        if((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower() -ne $digest){throw ('Existing preservation differs: '+$name)}
    }else{Copy-Item -LiteralPath $source -Destination $destination}
    if((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower() -ne $digest){throw ('Copy hash mismatch: '+$name)}
    $manifest[$name]=[ordered]@{bytes=(Get-Item -LiteralPath $source).Length;sha256=$digest}
}
$events=@(Get-WinEvent -FilterHashtable @{LogName='Application';Id=1000} -MaxEvents 25 -ErrorAction SilentlyContinue|Where-Object{$_.Message -match 'python.exe' -and $_.TimeCreated.ToUniversalTime() -ge [DateTime]::UtcNow.AddMinutes(-30)}|ForEach-Object{[ordered]@{created_utc=$_.TimeCreated.ToUniversalTime().ToString('o');record_id=$_.RecordId;values=@($_.Properties|ForEach-Object{[string]$_.Value})}})
$result=[ordered]@{passed=$true;checked_utc=[DateTime]::UtcNow.ToString('o');failed_pid=35364;supervisor_pid=44328;preserved_files=$manifest;missing_optional_files=$missingOptional;events=$events;scope='Preservation only; original raw/status/logs/budget copied and hashed. The missing wrapper exit file is explicit; terminal worker/parent evidence is STATUS, native traceback and absent WMI handles. No budget reconciliation or GPU relaunch.'}
$receipt=Join-Path $target 'PRESERVATION.json'
if(Test-Path -LiteralPath $receipt){throw 'Preservation receipt already exists; inspect instead of rerunning'}
$result|ConvertTo-Json -Depth 7|Set-Content -LiteralPath $receipt -Encoding UTF8
$result|ConvertTo-Json -Depth 7
Get-Content -LiteralPath (Join-Path $root 'logs/generation_validation_Ouro-2.6B.log') -Tail 45
