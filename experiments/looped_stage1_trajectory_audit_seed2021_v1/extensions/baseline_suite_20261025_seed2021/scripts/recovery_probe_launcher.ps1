$ErrorActionPreference='Stop'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
$start=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000
$p=Start-Process -FilePath 'D:\loopbase_env_20261025\Scripts\python.exe' -ArgumentList @('-u','-X','utf8',($root+'\scripts\recovery_boundary_replay.py')) -WindowStyle Hidden -RedirectStandardOutput ($root+'\logs\recovery_probe_stdout.log') -RedirectStandardError ($root+'\logs\recovery_probe_stderr.log') -PassThru
$timedOut=$false
try{Wait-Process -Id $p.Id -Timeout 600 -ErrorAction Stop}catch{$timedOut=$true;& taskkill.exe /PID $p.Id /T /F|Out-Null}
$p.Refresh();$code=if($timedOut){124}else{$p.ExitCode}
[ordered]@{pid=$PID;child_redirector_pid=$p.Id;started_unix=$start;finished_unix=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000;exit_code=$code;timeout=$timedOut}|ConvertTo-Json|Set-Content -LiteralPath ($root+'\logs\recovery_probe_exit.json') -Encoding UTF8
exit $code
