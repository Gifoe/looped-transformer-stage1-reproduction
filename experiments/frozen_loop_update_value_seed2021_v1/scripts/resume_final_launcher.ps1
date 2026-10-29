$ErrorActionPreference='Continue'
$root=Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8='1'
& 'D:/Codex_Research/environments/loopbase_env_20261025/Scripts/python.exe' -u -X utf8 ($root+'/scripts/resume_final.py') *> ($root+'/logs/final_resume_launcher.log')
@{exit_code=$LASTEXITCODE;finished_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath ($root+'/logs/final_exit.json') -Encoding UTF8
