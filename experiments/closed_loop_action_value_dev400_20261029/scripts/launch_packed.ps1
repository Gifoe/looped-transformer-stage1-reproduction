$ErrorActionPreference='Stop'
$root='D:/Codex_Research/experiments/closed_loop_action_value_dev400_20261029'
$env:PYTHONUTF8='1'
& 'D:/Codex_Research/environments/loopbase_env_20261025/Scripts/python.exe' -X utf8 -u "$root/scripts/supervise_packed.py" *> "$root/results/packed_launcher.log"
@{exit_code=$LASTEXITCODE} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath "$root/results/PACKED_LAUNCHER_EXIT.json"
