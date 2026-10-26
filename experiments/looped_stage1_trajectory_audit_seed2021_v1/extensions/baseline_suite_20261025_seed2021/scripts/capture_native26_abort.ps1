$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
$target=Join-Path $root 'logs\recovery_validation2p6_20261026'
$status=Get-Content -LiteralPath (Join-Path $root 'STATUS.json') -Raw|ConvertFrom-Json
$active=Get-Content -LiteralPath (Join-Path $root 'logs\active_gpu.json') -Raw|ConvertFrom-Json
if($status.stage -ne 'failed' -or $status.error -notlike '*3221225477*' -or $active.pid -ne 25608){throw 'Unexpected native-abort identity'}
$owners=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -eq 'python.exe' -and ($_.ProcessId -in @(25608,13640,27444) -or $_.CommandLine -like '*scripts/generate_math.py*' -or $_.CommandLine -like '*scripts/actual_halting.py*' -or $_.CommandLine -like '*scripts/run_suite.py*')})
if($owners.Count -gt 0){throw 'A possible experiment owner still exists; do not preserve/relaunch yet'}
[void](New-Item -ItemType Directory -Path $target -Force)
$names=@('STATUS.json','logs/active_gpu.json','logs/gpu_budget.json','logs/generation_validation_Ouro-2.6B.log','logs/math_validation_resume_stdout.log','logs/math_validation_resume_exit.json','results/math_prefix_audit_Ouro-2.6B.json','trajectories/math_Ouro-2.6B_validation.jsonl','results/model_load_Ouro-2.6B.json','logs/recovery_validation1p4_20261026/resume_controller_loaded_source.py')
$manifest=[ordered]@{}
foreach($name in $names){
    $source=Join-Path $root $name
    $destination=Join-Path $target ($name.Replace('/','__'))
    $digest=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower()
    if(Test-Path -LiteralPath $destination){
        if((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower() -ne $digest){throw ('Existing preservation differs: '+$name)}
    }else{Copy-Item -LiteralPath $source -Destination $destination}
    if((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower() -ne $digest){throw ('Copy hash mismatch: '+$name)}
    $manifest[$name]=[ordered]@{bytes=(Get-Item -LiteralPath $source).Length;sha256=$digest}
}
$events=@(Get-WinEvent -FilterHashtable @{LogName='Application';Id=1000;StartTime=[DateTime]::UtcNow.AddMinutes(-20)} -ErrorAction SilentlyContinue|Where-Object{$_.Message -match 'python.exe'}|Select-Object -First 6|ForEach-Object{
    $xml=[xml]$_.ToXml();$fields=[ordered]@{}
    foreach($field in $xml.Event.EventData.Data){$fields[[string]$field.Name]=[string]$field.'#text'}
    [ordered]@{created_utc=$_.TimeCreated.ToUniversalTime().ToString('o');fields=$fields}
})
$result=[ordered]@{passed=$true;checked_utc=[DateTime]::UtcNow.ToString('o');failed_pid=25608;supervisor_pid=27444;preserved_files=$manifest;events=$events;scope='Preservation only; original raw/status/logs/budget copied and hashed. No budget reconciliation, raw editing, source change or GPU relaunch.'}
$receipt=Join-Path $target 'PRESERVATION.json'
if(Test-Path -LiteralPath $receipt){throw 'Preservation receipt already exists; inspect instead of rerunning'}
$result|ConvertTo-Json -Depth 7|Set-Content -LiteralPath $receipt -Encoding UTF8
$result|ConvertTo-Json -Depth 7
Get-Content -LiteralPath (Join-Path $root 'logs/generation_validation_Ouro-2.6B.log') -Tail 45
