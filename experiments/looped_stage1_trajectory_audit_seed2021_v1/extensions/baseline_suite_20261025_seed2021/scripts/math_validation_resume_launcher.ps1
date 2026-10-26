$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
& 'E:\Anaconda\python.exe' -u -X utf8 ($root+'\scripts\run_suite.py') --phase math --resume-math-validation *> ($root+'\logs\math_validation_resume_stdout.log')
$code=$LASTEXITCODE
[ordered]@{exit_code=$code;finished_utc=[DateTime]::UtcNow.ToString('o');pid=$PID}|ConvertTo-Json|Set-Content -LiteralPath ($root+'\logs\math_validation_resume_exit.json') -Encoding UTF8
exit $code
