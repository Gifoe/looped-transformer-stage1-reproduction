$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1'
$env:PYTHONUTF8='1'
$p=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','utf8',($root+'\scripts\prepare_resources.py')) -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput ($root+'\logs\prepare_stdout.log') -RedirectStandardError ($root+'\logs\prepare_stderr.log') -PassThru -Wait
@{exit_code=$p.ExitCode; finished=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -Encoding UTF8 ($root+'\logs\prepare_process_exit.json')
