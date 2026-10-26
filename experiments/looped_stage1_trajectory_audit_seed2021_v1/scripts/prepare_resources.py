"""Pinned, resumable resource acquisition; stores all task state on D:."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading
import requests

ROOT=Path(__file__).resolve().parents[1]
for name in ['models','data','sources','logs','results','configs','tests','trajectories','figures']:
    (ROOT/name).mkdir(exist_ok=True)
os.environ['PIP_CACHE_DIR']=str(ROOT/'cache/pip')
os.environ['HF_HOME']=str(ROOT/'cache/huggingface')
SESSION=requests.Session(); SESSION.trust_env=False

def ranged_download(url,dest,total):
    part=dest.with_suffix(dest.suffix+'.partial')
    ledger=dest.with_suffix(dest.suffix+'.ranges.json')
    done=set(json.loads(ledger.read_text()) if ledger.exists() else [])
    if not part.exists() or part.stat().st_size!=total:
        with part.open('wb') as f:f.truncate(total)
        done=set()
    lock=threading.Lock();chunk=16*1024**2
    def one(index):
        if index in done:return
        lo=index*chunk;hi=min(total,(index+1)*chunk)-1
        for attempt in range(4):
            try:
                s=requests.Session();s.trust_env=False
                r=s.get(url,headers={'Range':f'bytes={lo}-{hi}'},timeout=(15,45))
                r.raise_for_status()
                if r.status_code!=206 or len(r.content)!=hi-lo+1:
                    raise RuntimeError(f'Invalid range response {r.status_code} {len(r.content)} expected {hi-lo+1}')
                if not r.headers.get('Content-Range','').startswith(f'bytes {lo}-{hi}/'):
                    raise RuntimeError('Incorrect range Content-Range')
                with part.open('r+b') as f:f.seek(lo);f.write(r.content)
                with lock:
                    done.add(index);ledger.write_text(json.dumps(sorted(done)))
                    if len(done)%16==0:print(json.dumps(dict(event='range_progress',path=str(dest),completed=len(done),total_chunks=(total+chunk-1)//chunk)),flush=True)
                return
            except Exception:
                if attempt==3:raise
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,range((total+chunk-1)//chunk)))
    part.replace(dest)
    return dict(path=str(dest),bytes=total,url=url,sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),transport='verified HTTP ranges')

def download(url, dest):
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size:
        return dict(path=str(dest),bytes=dest.stat().st_size,status='already_present')
    part=dest.with_suffix(dest.suffix+'.partial')
    if dest.suffix=='.safetensors':
        s=requests.Session();s.trust_env=False
        with s.get(url,headers={'Range':'bytes=0-1023'},timeout=(15,30),stream=True) as probe:
            if probe.status_code==206:
                total=int(probe.headers['Content-Range'].rsplit('/',1)[1])
                if total>128*1024**2:
                    if shutil.disk_usage(ROOT).free < total+5*1024**3:raise RuntimeError('Disk reserve would be violated')
                    return ranged_download(url,dest,total)
    for attempt in range(3):
        try:
            s=requests.Session(); s.trust_env=False
            offset=part.stat().st_size if part.exists() else 0
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with s.get(url,stream=True,timeout=(15,60),headers=headers) as r:
                r.raise_for_status()
                if r.status_code != 206: offset=0
                length=int(r.headers.get('Content-Length',0))
                if shutil.disk_usage(ROOT).free < length + 5*1024**3:
                    raise RuntimeError('Disk reserve would be violated')
                print(json.dumps(dict(event='download_start',url=url,bytes=length,offset=offset)),flush=True)
                with part.open('ab' if offset else 'wb') as f:
                    for chunk in r.iter_content(8*1024**2):
                        f.write(chunk)
            part.replace(dest)
            print(json.dumps(dict(event='download_done',path=str(dest),bytes=dest.stat().st_size)),flush=True)
            return dict(path=str(dest),bytes=dest.stat().st_size,url=url,sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest())
        except Exception as e:
            print(json.dumps(dict(event='download_error',url=url,attempt=attempt,error=str(e))),flush=True)
    return dict(url=url,error='download failed after 3 attempts')

def model(repo):
    r=SESSION.get('https://hf-mirror.com/api/models/'+repo,timeout=30); r.raise_for_status()
    meta=r.json(); rev=meta['sha']; directory=ROOT/'models'/repo.split('/')[-1]
    directory.mkdir(exist_ok=True)
    (directory/'snapshot_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    files=[x['rfilename'] for x in meta['siblings'] if x['rfilename'].endswith(('.json','.py','.safetensors','.txt')) and not x['rfilename'].startswith('assets/')]
    # Native safetensors only: no duplicate .bin weights, no optional Thinking model.
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(lambda f:download(f'https://hf-mirror.com/{repo}/resolve/{rev}/{f}?download=true',directory/f),files))
    result=dict(repo=repo,revision=rev,files=records)
    (directory/'download_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def install():
    venv=ROOT/'.venv'
    if not (venv/'Scripts/python.exe').exists():
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(venv)],check=True)
    py=venv/'Scripts/python.exe'
    packages=['numpy==2.2.6','transformers==4.57.1','huggingface-hub==0.36.0','accelerate==1.11.0','datasets==4.4.1','matplotlib==3.10.7','pandas==2.3.3','scipy==1.16.3','pytest==8.4.2']
    with (ROOT/'logs/dependency_install.log').open('w',encoding='utf-8') as log:
        r=subprocess.run([str(py),'-m','pip','install','--disable-pip-version-check','--index-url','https://pypi.tuna.tsinghua.edu.cn/simple',*packages],stdout=log,stderr=subprocess.STDOUT)
    result=dict(returncode=r.returncode,python=str(py),packages=packages)
    if r.returncode==0:
        with (ROOT/'logs/venv_smoke.log').open('w',encoding='utf-8') as log:
            x=subprocess.run([str(py),'-u','-c',"import torch,transformers,numpy; print(torch.__version__,transformers.__version__,numpy.__version__); print(torch.cuda.get_device_name(0)); x=torch.ones((4,4),device='cuda',dtype=torch.bfloat16); print(x@x)"],stdout=log,stderr=subprocess.STDOUT)
        result['smoke_returncode']=x.returncode
    (ROOT/'logs/dependency_status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    results={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={'environment':pool.submit(install),'ouro':pool.submit(model,'ByteDance/Ouro-1.4B'),'qwen':pool.submit(model,'Qwen/Qwen3-4B-Base')}
        for key,job in jobs.items():
            try: results[key]=job.result()
            except Exception as e: results[key]=dict(error=repr(e))
    (ROOT/'logs/resource_status.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results),flush=True)
