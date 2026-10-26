# Relocate only the extension-added Ouro-2.6B snapshot. Existing Stage1 is sealed.
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
$models='D:\looped_stage1_trajectory_audit_seed2021_v1\models'
$source=Join-Path $models 'Ouro-2.6B'
$backup=Join-Path $models 'Ouro-2.6B.relocation_verified_backup_20261026'
$storage='E:\loopbase_20261025_seed2021'
$destination=Join-Path $storage 'models\Ouro-2.6B'
$evidence=Join-Path $root 'logs\storage_migration_20261026.json'
$report=[ordered]@{passed=$false;started_utc=[DateTime]::UtcNow.ToString('o');pid=$PID;source=$source;destination=$destination;backup=$backup;stage='preflight';files=@();authorization='User explicitly authorized other disks with clearly listed folders';inference_code_unchanged=$true}
function Save-Progress([string]$stage){
    $report.stage=$stage;$report.updated_utc=[DateTime]::UtcNow.ToString('o')
    $report|ConvertTo-Json -Depth 9|Set-Content -LiteralPath $evidence -Encoding UTF8
    Write-Output ($stage+' '+$report.updated_utc)
}
function Assert-NoModelWorker {
    $workers=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python' -and $_.CommandLine -match '--model[ =]+Ouro-2\.6B'})
    if($workers.Count){throw 'Ouro-2.6B worker is active; do not relocate its files'}
}
try{
    if([IO.Path]::GetFullPath($source) -ne ($models+'\Ouro-2.6B')){throw 'Unexpected source path'}
    if([IO.Path]::GetFullPath($backup) -ne ($models+'\Ouro-2.6B.relocation_verified_backup_20261026')){throw 'Unexpected backup path'}
    if([IO.Path]::GetFullPath($destination) -ne 'E:\loopbase_20261025_seed2021\models\Ouro-2.6B'){throw 'Unexpected destination path'}
    $original=Get-Item -LiteralPath $source
    if($original.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Source already has a link; inspect before retry'}
    if((Test-Path -LiteralPath $destination) -or (Test-Path -LiteralPath $backup)){throw 'Destination or backup exists; inspect before retry'}
    if(@(Get-ChildItem -LiteralPath $source -Recurse -Force|Where-Object{$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count){throw 'Nested model link is not supported'}
    Assert-NoModelWorker
    $sourceFiles=@(Get-ChildItem -LiteralPath $source -Recurse -File -Force)
    $bytes=($sourceFiles|Measure-Object -Property Length -Sum).Sum
    if((Get-PSDrive E).Free -lt ($bytes+10GB)){throw 'E drive has insufficient reserve'}
    $report.bytes=$bytes;$report.before_d_free_bytes=(Get-PSDrive D).Free;$report.before_e_free_bytes=(Get-PSDrive E).Free
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force|Out-Null
    Save-Progress 'copying'
    Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
    Save-Progress 'verifying_copy'
    $verified=@(foreach($file in $sourceFiles){
        $relative=$file.FullName.Substring($source.Length+1)
        $target=Join-Path $destination $relative
        $copy=Get-Item -LiteralPath $target
        if($copy.Length -ne $file.Length){throw ('Size mismatch: '+$relative)}
        $a=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower()
        $b=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower()
        if($a -ne $b){throw ('SHA256 mismatch: '+$relative)}
        [pscustomobject]@{relative_path=$relative;bytes=$file.Length;sha256=$a;copy_sha256=$b;passed=$true}
    })
    if(@(Get-ChildItem -LiteralPath $destination -Recurse -File -Force).Count -ne $sourceFiles.Count){throw 'Copy file count differs'}
    $report.files=$verified
    Save-Progress 'copy_verified'
    Assert-NoModelWorker
    Rename-Item -LiteralPath $source -NewName (Split-Path -Leaf $backup)
    try{
        New-Item -ItemType Junction -Path $source -Target $destination|Out-Null
        $link=Get-Item -LiteralPath $source -Force
        if($link.LinkType -ne 'Junction' -or @($link.Target)[0] -ne $destination){throw 'Junction target mismatch'}
        foreach($entry in $verified){if((Get-Item -LiteralPath (Join-Path $source $entry.relative_path)).Length -ne $entry.bytes){throw 'Logical model path verification failed'}}
        $revision=(Get-Content -LiteralPath (Join-Path $source 'snapshot_metadata.json') -Raw|ConvertFrom-Json).sha
        if($revision -ne '1ed04250da1a9936042725d302e81c8fa2ab5abd'){throw 'Pinned revision differs through junction'}
    }catch{
        if(Test-Path -LiteralPath $source){[IO.Directory]::Delete($source,$false)}
        Rename-Item -LiteralPath $backup -NewName 'Ouro-2.6B'
        throw
    }
    Save-Progress 'junction_verified'
    # Delete only the exact checked backup of the newly downloaded model, after
    # every copy hash passed and its replacement logical path was verified.
    $resolvedBackup=(Resolve-Path -LiteralPath $backup).Path
    if($resolvedBackup -ne ($models+'\Ouro-2.6B.relocation_verified_backup_20261026')){throw 'Backup delete path differs'}
    if((Get-Item -LiteralPath $backup).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Backup unexpectedly became a link'}
    Remove-Item -LiteralPath $backup -Recurse -Force
    $report.after_d_free_bytes=(Get-PSDrive D).Free;$report.after_e_free_bytes=(Get-PSDrive E).Free
    $report.link_type=(Get-Item -LiteralPath $source).LinkType
    Save-Progress 'writing_layout'
    $layout=@"
# Server storage layout

Updated UTC: $($report.updated_utc)

| Purpose | Actual server path |
| --- | --- |
| Sealed Stage1 and existing Qwen/Ouro1.4 snapshots | D:\looped_stage1_trajectory_audit_seed2021_v1 |
| Extension scripts, configs, raw results, logs and reports | $root |
| Extension Python environment | D:\loopbase_env_20261025 |
| Newly added Ouro-2.6B snapshot, physical location | $destination |
| Compatible original model path, NTFS junction to E | $source |
| F drive | Inspected, approximately 218 GiB free at preflight; no project files stored there |

All $($verified.Count) snapshot files were copied with identical SHA256 hashes.
Migration evidence: logs/storage_migration_20261026.json.
Existing Stage1 files, model bytes, protocol and inference code were unchanged.
The junction preserves model loading paths; it does not allocate a second copy.
No currently active Ouro-2.6B worker existed during the migration.
"@
    $layout|Set-Content -LiteralPath (Join-Path $root 'STORAGE_LAYOUT.md') -Encoding UTF8
    $layout|Set-Content -LiteralPath (Join-Path $storage 'STORAGE_LAYOUT.md') -Encoding UTF8
    $report.passed=$true
    Save-Progress 'completed'
}catch{
    $report.error=$_.Exception.ToString();Save-Progress 'failed';throw
}
