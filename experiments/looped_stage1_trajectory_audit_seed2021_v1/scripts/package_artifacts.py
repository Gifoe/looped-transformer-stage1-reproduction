"""Verify completeness and make a small source/result delivery; excludes weights."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
audit={};ids={}
for name,count in [('ouro_validation',100),('ouro_test',300),('ouro_native_gate',400),('ouro_math_teacherforced',100),('math_generation',400),('math_depth_and_hidden',400)]:
    records=read(ROOT/'trajectories'/(name+'.jsonl'))
    if name.startswith('math_'):
        keys={(r['id'],r['method']) for r in records}
        methods=sorted({r['method'] for r in records})
        assert all(sum(r['method']==m for r in records)==100 for m in methods)
        ids[name]={r['id'] for r in records}
        assert len(ids[name])==100
        assert all({r['id'] for r in records if r['method']==m}==ids[name] for m in methods)
    else:keys={r['id'] for r in records};ids[name]=keys
    assert len(records)==len(keys)==count,(name,len(records),len(keys))
    audit[name]=dict(records=count,unique=True)
assert not ids['ouro_test']&ids['ouro_validation']
assert ids['ouro_native_gate']==ids['ouro_test']|ids['ouro_validation']
assert ids['math_generation']==ids['math_depth_and_hidden']
assert ids['ouro_math_teacherforced']==ids['math_generation']
for method in ['A0','A1','A2','A3','A4']:
    for split,count in [('pilot',50),('test',600)]:
        records=read(ROOT/'results'/('qwen_'+method+'_'+split+'.jsonl'));keys={r['id'] for r in records}
        assert len(records)==len(keys)==count
        ids[method+'_'+split]=keys;audit[method+'_'+split]=dict(records=count,unique=True)
    assert ids[method+'_test']==ids['A0_test']
    assert not ids[method+'_test']&ids[method+'_pilot']
assert ids['ouro_test']<=ids['A0_test']
for name in ['ouro_parity.json','qwen_parity.json','generation_parity.json','math_depth_parity.json','math_teacherforced_parity.json','scoring_parity.json','native_gate_parity.json','checkpoint_integrity.json']:
    assert json.loads((ROOT/'results'/name).read_text())['passed'],name
    audit[name]='passed'
budget=json.loads((ROOT/'logs/gpu_budget.json').read_text());assert budget['used_seconds']<=50*3600
audit['gpu_process_wall_hours_including_smoke_reserve']=budget['used_seconds']/3600
audit['scope_limitation']='Independent Qwen recipe, zero-shot supplementary MC protocol; direct author-code parity unavailable. Halting is offline only.'
model_meta=ROOT/'results/model_provenance';model_meta.mkdir(exist_ok=True)
for model in ['Ouro-1.4B','Qwen3-4B-Base']:
    for name in ['config.json','snapshot_metadata.json','download_manifest.json','modeling_ouro.py','configuration_ouro.py']:
        source=ROOT/'models'/model/name
        if source.exists():shutil.copy2(source,model_meta/(model+'__'+name))
subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(ROOT/'logs/pip_freeze.txt').open('w',encoding='utf-8'),check=True)
audit['project_disk_bytes']=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file() and f.suffix!='.zip')
audit['d_free_bytes']=shutil.disk_usage(ROOT).free
audit['created_unix']=time.time()
audit['delivery_cache_note']='Complete dataset parquet snapshots and model weights remain on server D drive; portable delivery retains manifests, source revisions and all evaluated samples/results.'
(ROOT/'results/completion_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
files=[f for f in ROOT.rglob('*') if f.is_file() and not set(f.relative_to(ROOT).parts)&{'.venv','models','cache','.git','__pycache__','transport','.pytest_cache'} and f.suffix not in ['.zip','.tmp','.pyc'] and f.name!='ARTIFACT_MANIFEST.json']
files=[f for f in files if not f.is_relative_to(ROOT/'data/snapshots')]
manifest={f.relative_to(ROOT).as_posix():dict(bytes=f.stat().st_size,sha256=digest(f)) for f in files}
(ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with zipfile.ZipFile(ROOT/'looped_stage1_delivery.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for f in files+[ROOT/'ARTIFACT_MANIFEST.json']:archive.write(f,f.relative_to(ROOT).as_posix())
print(json.dumps(dict(archive=str(ROOT/'looped_stage1_delivery.zip'),bytes=(ROOT/'looped_stage1_delivery.zip').stat().st_size,audit=audit)),flush=True)
