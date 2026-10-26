$ErrorActionPreference='Stop'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
$p=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','utf8',($root+'\scripts\prepare_suite.py')) -WorkingDirectory $root -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput ($root+'\logs\prepare_stdout.log') -RedirectStandardError ($root+'\logs\prepare_stderr.log')
@{process_id=$p.Id;exit_code=$p.ExitCode;finished_utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath ($root+'\logs\prepare_exit.json') -Encoding UTF8
