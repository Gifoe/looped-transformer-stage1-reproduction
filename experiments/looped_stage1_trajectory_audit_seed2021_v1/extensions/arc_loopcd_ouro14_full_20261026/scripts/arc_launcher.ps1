$ErrorActionPreference='Continue'
$env:PYTHONUTF8='1'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\arc_loopcd_ouro14_full_20261026'
if(Test-Path -LiteralPath ($root+'\logs\arc_launcher_exit.json')){throw 'Historical launcher receipt exists; inspect before repeating'}
& 'E:\Anaconda\python.exe' -u -X utf8 -X faulthandler ($root+'\scripts\arc_controller.py') *> ($root+'\logs\arc_launcher.log')
$code=$LASTEXITCODE
[ordered]@{exit_code=$code;finished_utc=[DateTime]::UtcNow.ToString('o');launcher_pid=$PID}|ConvertTo-Json|Set-Content -LiteralPath ($root+'\logs\arc_launcher_exit.json') -Encoding UTF8
exit $code
