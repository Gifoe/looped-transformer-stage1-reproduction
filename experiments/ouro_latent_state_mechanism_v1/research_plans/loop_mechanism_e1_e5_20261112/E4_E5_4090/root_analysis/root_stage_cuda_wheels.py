"""Stage two pinned CUDA wheels via official PyPI and resumable parallel transfer."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
import paramiko

LOCAL = Path('D:/ouro-e4-e5-20261112/wheelhouse')
STATE = Path('D:/ouro-e4-e5-20261112/results/cuda_wheel_staging.json')
PACKAGES = [
    ('nvidia-cudnn-cu12','9.1.0.70','nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl',664752741,'165764f44ef8c61fcdfdfdbe769d687e06374059fbb388b6c89ecb0e28793a6f'),
    ('nvidia-cublas-cu12','12.4.5.8','nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl',363438805,'2fc8da60df463fdefa81e323eef2e36489e1c94335b5358bcb38360adf75ac9b'),
]
progress = {}
lock = threading.Lock()


def update(name, **kw):
    with lock:
        progress.setdefault(name, {}).update(kw)


def ssh():
    c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('connect.bjb1.seetacloud.com',port=40321,username='root',password=os.environ['OURO_TARGET_PASSWORD'],timeout=20,allow_agent=False,look_for_keys=False)
    c.get_transport().set_keepalive(20)
    return c


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def stage(package):
    pkg,version,name,total,expected=package
    with urllib.request.urlopen('https://pypi.org/pypi/'+pkg+'/'+version+'/json',timeout=20) as r:meta=json.load(r)
    artifact=next(x for x in meta['urls'] if x['filename']==name)
    assert artifact['digests']['sha256']==expected and artifact['size']==total
    url=artifact['url'];path=LOCAL/name;partial=path.with_suffix('.whl.partial')
    update(name,status='DOWNLOADING',total=total,downloaded=0,uploaded=0)
    if not path.exists():
        block=4*1024*1024;ranges=[(i,min(i+block,total)-1) for i in range(0,total,block)]
        with partial.open('wb') as f:f.truncate(total)
        def download(pair):
            start,end=pair
            for attempt in range(4):
                try:
                    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req,timeout=40) as r:
                        assert r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{total}'
                        data=r.read(end-start+2)
                    assert len(data)==end-start+1
                    with partial.open('r+b') as f:f.seek(start);f.write(data)
                    with lock:progress[name]['downloaded']+=len(data)
                    return
                except Exception:
                    if attempt==3:raise
        with cf.ThreadPoolExecutor(max_workers=8) as p:list(p.map(download,ranges))
        assert digest(partial)==expected
        partial.replace(path)
    assert path.stat().st_size==total and digest(path)==expected
    update(name,status='UPLOADING',downloaded=total)
    remote='/root/autodl-tmp/ouro-e4-e5/root_wheel_staging/'+name
    c=ssh()
    _,o,e=c.exec_command('mkdir -p '+remote+' /root/autodl-tmp/ouro-e4-e5/wheelhouse');assert o.channel.recv_exit_status()==0,e.read().decode();c.close()
    part_progress=[0]*4
    def upload(i):
        start,end=total*i//4,total*(i+1)//4
        for attempt in range(4):
            c=ssh()
            try:
                s=c.open_sftp();rp=remote+'/part%02d'%i
                try:exists=s.stat(rp).st_size
                except FileNotFoundError:exists=0
                assert exists<=end-start
                with path.open('rb') as f:
                    f.seek(start+exists)
                    with s.open(rp,'ab') as dst:
                        dst.set_pipelined(True)
                        while exists<end-start:
                            data=f.read(min(1024*1024,end-start-exists));assert data
                            dst.write(data);exists+=len(data)
                            with lock:
                                part_progress[i]=exists;progress[name]['uploaded']=sum(part_progress)
                assert s.stat(rp).st_size==end-start
                return
            except Exception:
                if attempt==3:raise
            finally:c.close()
    with cf.ThreadPoolExecutor(max_workers=4) as p:list(p.map(upload,range(4)))
    code="""from pathlib import Path
import hashlib,os,json
root=Path(REMOTE)
dest=Path('/root/autodl-tmp/ouro-e4-e5/wheelhouse')/NAME
tmp=dest.with_suffix('.whl.root-staging.partial');h=hashlib.sha256()
with tmp.open('wb') as out:
 for i in range(4):
  with (root/('part%02d'%i)).open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b);out.write(b)
assert h.hexdigest()==EXPECTED,h.hexdigest()
assert not dest.exists(),'refuse to overwrite existing wheel'
os.replace(tmp,dest)
print(json.dumps({'file':NAME,'bytes':dest.stat().st_size,'sha256':h.hexdigest()}))
"""
    code='REMOTE='+repr(remote)+'\nNAME='+repr(name)+'\nEXPECTED='+repr(expected)+'\n'+code
    c=ssh()
    try:
        inp,o,e=c.exec_command('/root/miniconda3/bin/python -',timeout=90);inp.write(code);inp.channel.shutdown_write();output=o.read().decode();error=e.read().decode();assert o.channel.recv_exit_status()==0,error
        update(name,status='VERIFIED',downloaded=total,uploaded=total,receipt=json.loads(output))
    finally:c.close()


def main():
    LOCAL.mkdir(parents=True,exist_ok=True);started=time.time()
    with cf.ThreadPoolExecutor(max_workers=2) as p:
        jobs=[p.submit(stage,x) for x in PACKAGES]
        while not all(j.done() for j in jobs):
            with lock:row={'seconds':time.time()-started,'packages':json.loads(json.dumps(progress))}
            STATE.write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True);time.sleep(10)
        for j in jobs:j.result()
    row={'seconds':time.time()-started,'packages':progress,'status':'BOTH_VERIFIED'}
    STATE.write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)


if __name__=='__main__':main()
