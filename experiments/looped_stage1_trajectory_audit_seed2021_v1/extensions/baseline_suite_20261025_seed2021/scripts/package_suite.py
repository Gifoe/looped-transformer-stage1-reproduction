"""Seal only the extension; never run or modify the old Stage1 packager."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
from common import ROOT,BASE

EXCLUDED={'.git','.venv','cache','__pycache__','transport','delivery'}
def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):result.update(block)
    return result.hexdigest()

def main():
    assert json.loads((ROOT/'results/completion_audit.json').read_text())['passed']
    assert json.loads((ROOT/'STATUS.json').read_text())['stage']=='completed'
    assert (ROOT/'FINDINGS.md').exists() and (ROOT/'REPRODUCE.md').exists()
    assert shutil.disk_usage(ROOT).free>5*1024**3
    manifest={}
    for directory,subdirectories,files in os.walk(ROOT):
        # Prune before traversal, including the retained failed long-path venv.
        # Filtering rglob results afterwards would still visit every cache/env
        # file and can hit the original Windows path-length failure again.
        subdirectories[:]=sorted(name for name in subdirectories if name not in EXCLUDED)
        for filename in sorted(files):
            path=Path(directory)/filename;relative=path.relative_to(ROOT)
            if relative.name in ['ARTIFACT_MANIFEST.json','DELIVERY.json'] or relative.suffix=='.tmp':continue
            if relative.parts[0]=='logs' and relative.name.startswith('final_supervisor'):continue
            manifest[relative.as_posix()]=dict(bytes=path.stat().st_size,sha256=digest(path))
    (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
    directory=ROOT/'delivery';directory.mkdir(exist_ok=True)
    target=directory/'baseline_suite_20261025_seed2021.zip';temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for name,entry in manifest.items():
            path=ROOT/name;assert digest(path)==entry['sha256'],name
            archive.write(path,name)
        archive.write(ROOT/'ARTIFACT_MANIFEST.json','ARTIFACT_MANIFEST.json')
    temporary.replace(target)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name,entry in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==entry['sha256'],name
    metadata=dict(zip_name=target.name,bytes=target.stat().st_size,sha256=digest(target),files=len(manifest),manifest_sha256=digest(ROOT/'ARTIFACT_MANIFEST.json'),stage1_manifest_sha256=digest(BASE/'ARTIFACT_MANIFEST.json'),excluded=['model weights','dataset snapshot parquet','Python environments','cache','transport credentials and transient transfer logs'],install_relative_to_stage1='extensions/baseline_suite_20261025_seed2021')
    (ROOT/'DELIVERY.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
