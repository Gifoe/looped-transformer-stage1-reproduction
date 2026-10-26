$ErrorActionPreference='Stop'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
$p=Start-Process -FilePath 'D:\loopbase_env_20261025\Scripts\python.exe' -ArgumentList @('-u','-X','utf8',($root+'\scripts\prepare_data.py')) -WorkingDirectory $root -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput ($root+'\logs\data_stdout.log') -RedirectStandardError ($root+'\logs\data_stderr.log')
@{process_id=$p.Id;exit_code=$p.ExitCode;finished_utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath ($root+'\logs\data_exit.json') -Encoding UTF8
