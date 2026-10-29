$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$stage=$args[0]
if($stage -notin @('pilot','discovery','head_train','validation','final_sciq','final_piqa')){throw 'Unknown stage'}
if(Test-Path -LiteralPath ($root+'/logs/'+$stage+'_exit.json')){throw 'Existing stage receipt; inspect before rerun'}
$env:PYTHONUTF8='1'
& 'D:/Codex_Research/environments/loopbase_env_20261025/Scripts/python.exe' -u -X utf8 -X faulthandler ($root+'/scripts/supervise.py') $stage *> ($root+'/logs/'+$stage+'_launcher.log')
$code=$LASTEXITCODE
@{exit_code=$code;finished_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath ($root+'/logs/'+$stage+'_launcher_exit.json') -Encoding UTF8
exit $code
