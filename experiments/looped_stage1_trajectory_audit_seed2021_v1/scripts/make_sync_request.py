"""Local-only helper: construct SFTP request without shell interpolation."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
files=[f for sub in ['scripts','configs','tests'] for f in (ROOT/sub).rglob('*') if f.is_file() and '__pycache__' not in f.parts]
files += [ROOT/name for name in ['README.md','TASK.md','.gitignore','REPRODUCE.md','ISSUE_LOG.md'] if (ROOT/name).exists()]
files += [f for f in (ROOT/'sources').glob('*') if f.is_file()]
kind='finalization' if '--finalization' in sys.argv else ('extension' if '--extension' in sys.argv else 'pipeline')
request=dict(upload=[[str(f),'D:/looped_stage1_trajectory_audit_seed2021_v1/'+f.relative_to(ROOT).as_posix()] for f in files],
    powershell="$ProgressPreference='SilentlyContinue'; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\\looped_stage1_trajectory_audit_seed2021_v1\\scripts\\"+kind+"_launcher.ps1'} | Select-Object ProcessId,ReturnValue | ConvertTo-Json -Compress",
    save_output=str(ROOT/'logs'/(kind+'_launch.json')))
if '--upload-only' in sys.argv:
    request.pop('powershell',None);request.pop('save_output',None)
    filename='sync_upload_only.json'
else:filename='sync_and_launch_'+kind+'.json'
(ROOT/'transport'/filename).write_text(json.dumps(request),encoding='utf-8')
print(f'{len(files)} files queued')
