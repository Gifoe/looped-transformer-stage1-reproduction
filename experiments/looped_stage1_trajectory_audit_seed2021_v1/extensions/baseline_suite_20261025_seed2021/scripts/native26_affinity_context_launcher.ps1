$ErrorActionPreference='Continue'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
& 'E:\Anaconda\python.exe' -u -X utf8 -X faulthandler ($root+'\scripts\native26_affinity_context_controller.py') *> ($root+'\logs\native26_affinity_context_controller.log')
$code=$LASTEXITCODE
$result=[ordered]@{exit_code=$code;finished_utc=[DateTime]::UtcNow.ToString('o');launcher_pid=$PID}
[IO.File]::WriteAllText(($root+'\logs\native26_affinity_context_exit.json'),($result|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
exit $code
