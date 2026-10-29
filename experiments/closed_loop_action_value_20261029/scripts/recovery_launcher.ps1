$ErrorActionPreference='Stop'
$root='D:/Codex_Research/experiments/closed_loop_action_value_20261029'
$env:PYTHONUTF8='1'
& 'D:/Codex_Research/environments/loopbase_env_20261025/Scripts/python.exe' -X utf8 -u "$root/scripts/supervise_pilot.py" *> "$root/results/supervisor_launcher.log"
@{exit_code=$LASTEXITCODE} | ConvertTo-Json | Set-Content -LiteralPath "$root/results/RECOVERY_LAUNCHER_EXIT.json" -Encoding UTF8
