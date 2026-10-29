$ErrorActionPreference = 'Stop'
$root = 'D:/Codex_Research/experiments/closed_loop_action_value_20261029'
Set-Location -LiteralPath $root
$start = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
& 'D:/Codex_Research/environments/loopbase_env_20261025/Scripts/python.exe' -X utf8 -u "$root/scripts/pilot.py" *> "$root/results/pilot.log"
$code = $LASTEXITCODE
@{exit_code=$code;start_unix=$start;end_unix=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath "$root/results/EXIT.json"
exit $code
