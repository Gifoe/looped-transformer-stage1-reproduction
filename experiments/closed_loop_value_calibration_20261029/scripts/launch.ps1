$ErrorActionPreference='Stop'
$root='D:/Codex_Research/experiments/closed_loop_value_calibration_20261029'
$env:PYTHONUTF8='1'
& 'D:/Codex_Research/environments/loopbase_env_20261025/Scripts/python.exe' -X utf8 -u "$root/scripts/supervise.py" *> "$root/results/launcher.log"
@{exit_code=$LASTEXITCODE} | ConvertTo-Json | Set-Content -LiteralPath "$root/results/LAUNCHER_EXIT.json" -Encoding UTF8
