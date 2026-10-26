"""CPU resource preparation in an isolated extension environment."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
CFG=json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf-8'))
for folder in ['sources','data','results','logs','configs','tests','scripts','trajectories','figures']:(ROOT/folder).mkdir(exist_ok=True)
os.environ['PIP_CACHE_DIR']=str(ROOT/'cache/pip')
os.environ['HF_HOME']=str(ROOT/'cache/huggingface')
os.environ['HF_ENDPOINT']='https://hf-mirror.com'

def install():
    py=Path(CFG['environment_python']);env=py.parents[1]
    if not py.exists():subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(env)],check=True)
    # Reuse the pinned Stage1 wheels without mutating that environment. This
    # also keeps its pip NumPy ahead of base Anaconda's incompatible MKL wheel.
    (env/'Lib/site-packages/stage1_shared.pth').write_text(str(BASE/'.venv/Lib/site-packages')+'\n',encoding='utf-8')
    with (ROOT/'logs/environment_install.log').open('a',encoding='utf-8') as log:
        command=[str(py),'-m','pip','install','--disable-pip-version-check','--index-url','https://pypi.tuna.tsinghua.edu.cn/simple','-c',str(ROOT/'configs/env_constraints.txt'),'lm_eval==0.4.9','peft==0.17.1']
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    code="import lm_eval,numpy,transformers,importlib.metadata as m;print(m.version('lm_eval'),numpy.__version__,transformers.__version__);from lm_eval.api.task import ConfigurableTask;from lm_eval.models.huggingface import HFLM;print('HARNESS_IMPORT_PASSED')"
    with (ROOT/'logs/environment_import.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(py),'-u','-X','utf8','-c',code],stdout=log,stderr=subprocess.STDOUT,check=True)
    with (ROOT/'logs/pip_freeze.txt').open('w',encoding='utf-8') as log:subprocess.run([str(py),'-m','pip','freeze'],stdout=log,check=True)
    return dict(passed=True,python=str(py),shared_readonly_site_packages=str(BASE/'.venv/Lib/site-packages'))

def model():
    sys.path.insert(0,str(BASE/'scripts'))
    from prepare_resources import download
    os.environ['PIP_CACHE_DIR']=str(ROOT/'cache/pip')
    os.environ['HF_HOME']=str(ROOT/'cache/huggingface')
    import requests
    session=requests.Session();session.trust_env=False
    response=session.get('https://hf-mirror.com/api/models/ByteDance/Ouro-2.6B?blobs=true',timeout=30);response.raise_for_status()
    meta=response.json();revision=CFG['ouro26_revision'];assert meta['sha']==revision
    selected=[x for x in meta['siblings'] if x['rfilename'].endswith(('.json','.py','.safetensors','.txt')) and not x['rfilename'].startswith('assets/')]
    remaining=sum(x.get('size',0) for x in selected if not (BASE/'models/Ouro-2.6B'/x['rfilename']).exists())
    if shutil.disk_usage(ROOT).free < remaining+CFG['free_disk_reserve_gib']*1024**3:raise RuntimeError('Insufficient disk reserve for new model')
    directory=BASE/'models/Ouro-2.6B';directory.mkdir(exist_ok=True)
    (directory/'snapshot_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    records=[]
    # One large shard only; its helper already uses four verified HTTP ranges.
    for item in selected:
        name=item['rfilename'];result=download(f'https://hf-mirror.com/ByteDance/Ouro-2.6B/resolve/{revision}/{name}?download=true',directory/name)
        if 'error' in result:raise RuntimeError(result)
        if name.endswith('.safetensors'):
            with (directory/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            expected=item['lfs']['sha256'];assert actual==expected,(name,actual,expected)
            result.update(expected_sha256=expected,actual_sha256=actual,passed=True)
        records.append(result)
    report=dict(passed=True,repo='ByteDance/Ouro-2.6B',revision=revision,files=records)
    (directory/'download_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'results/ouro26_checkpoint_integrity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    outcomes={};errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={'environment':pool.submit(install),'model':pool.submit(model)}
        for key,future in pending.items():
            try:outcomes[key]=future.result()
            except Exception as exc:outcomes[key]=dict(error=repr(exc));errors.append(key)
    outcomes['completed_unix']=time.time();outcomes['passed']=not errors
    (ROOT/'logs/resource_status.json').write_text(json.dumps(outcomes,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=not errors,errors=errors)),flush=True)
    if errors:raise RuntimeError('Preparation failed: '+','.join(errors))
