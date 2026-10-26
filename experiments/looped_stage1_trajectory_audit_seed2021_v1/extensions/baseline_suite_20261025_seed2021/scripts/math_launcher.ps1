$ErrorActionPreference='Stop'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
# The prior supervisor owns the GPU until it actually exits. A STATUS snapshot
# is never sufficient evidence to launch a competing worker.
$prior=Get-CimInstance Win32_Process -Filter 'ProcessId=26836'
if($prior){
  if($prior.CommandLine -notlike '*run_suite.py*--phase mc*'){throw 'Prior PID was reused; investigate instead of starting another GPU owner'}
  Wait-Process -Id 26836 -ErrorAction SilentlyContinue
}
$deadline=[DateTime]::UtcNow.AddSeconds(30)
while(!(Test-Path -LiteralPath ($root+'\logs\mc_supervisor_exit.json')) -and [DateTime]::UtcNow -lt $deadline){Start-Sleep -Seconds 1}
$priorExit=Get-Content -LiteralPath ($root+'\logs\mc_supervisor_exit.json') -Raw|ConvertFrom-Json
if($priorExit.exit_code -ne 0){
  $recovery=Get-Content -LiteralPath ($root+'\results\mc_recovery_gate.json') -Raw|ConvertFrom-Json
  if(!$recovery.passed -or !$recovery.all_gpu_outputs_preserved){throw 'MC failed without verified CPU-only recovery'}
}
foreach($name in @('mc_selected.json','halting_selected.json')){if(!(Test-Path -LiteralPath ($root+'\configs\'+$name))){throw ('Missing MC analysis output: '+$name)}}
$p=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','utf8',($root+'\scripts\run_suite.py'),'--phase','math') -WorkingDirectory $root -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput ($root+'\logs\math_supervisor_stdout.log') -RedirectStandardError ($root+'\logs\math_supervisor_stderr.log')
@{process_id=$p.Id;exit_code=$p.ExitCode;finished_utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath ($root+'\logs\math_supervisor_exit.json') -Encoding UTF8
