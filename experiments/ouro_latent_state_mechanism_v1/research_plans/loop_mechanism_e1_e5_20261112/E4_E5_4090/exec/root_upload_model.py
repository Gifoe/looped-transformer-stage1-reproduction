"""Resumable eight-stream upload of an existing, independently hashed checkpoint."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import paramiko

SOURCE = Path('D:/chenyu-iclr/experiments/ouro_weight_space_repair_v1/models/Ouro-1.4B/model.safetensors')
REMOTE = '/root/autodl-tmp/ouro-e4-e5/local_model_transfer'
EXPECTED = '58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
PROGRESS = Path('D:/ouro-e4-e5-20261112/results/model_upload_progress.json')
N = 8
progress = [0] * N
lock = threading.Lock()


def client():
    c = paramiko.SSHClient()
    c.load_system_host_keys()
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('connect.bjb1.seetacloud.com', port=40321, username='root',
              password=os.environ['OURO_TARGET_PASSWORD'], timeout=20,
              allow_agent=False, look_for_keys=False)
    c.get_transport().set_keepalive(20)
    return c


def upload(i, total):
    start, end = total * i // N, total * (i+1) // N
    path = REMOTE + '/part%02d' % i
    for attempt in range(4):
        c = client()
        try:
            s = c.open_sftp()
            try:
                existing = s.stat(path).st_size
            except FileNotFoundError:
                existing = 0
            assert existing <= end-start
            with lock:
                progress[i] = existing
            with SOURCE.open('rb') as src:
                src.seek(start+existing)
                with s.open(path, 'ab') as dst:
                    dst.set_pipelined(True)
                    while existing < end-start:
                        data = src.read(min(1024*1024, end-start-existing))
                        assert data
                        dst.write(data)
                        existing += len(data)
                        with lock:
                            progress[i] = existing
            assert s.stat(path).st_size == end-start
            return i
        except Exception as exc:
            print(json.dumps({'part':i,'attempt':attempt,'error':str(exc)}), flush=True)
            if attempt == 3:
                raise
        finally:
            c.close()


def main():
    c = client()
    _, out, err = c.exec_command('mkdir -p ' + REMOTE + ' /root/autodl-tmp/ouro-e4-e5/model')
    assert out.channel.recv_exit_status() == 0, err.read().decode()
    c.close()
    total = SOURCE.stat().st_size
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
        futures = [pool.submit(upload, i, total) for i in range(N)]
        while not all(f.done() for f in futures):
            with lock:
                done = sum(progress)
            row = {'status':'UPLOADING','bytes':done,'total':total,'fraction':done/total,
                   'seconds':time.time()-started,'average_MiB_s':done/1048576/max(time.time()-started,.001)}
            tmp = PROGRESS.with_suffix('.tmp')
            tmp.write_text(json.dumps(row))
            tmp.replace(PROGRESS)
            print(json.dumps(row), flush=True)
            time.sleep(10)
        for f in futures:
            f.result()
    c = client()
    code = '''from pathlib import Path
import hashlib,os,json
root=Path('/root/autodl-tmp/ouro-e4-e5')
dest=root/'model/model.safetensors'
tmp=dest.with_suffix('.safetensors.local_upload.partial')
h=hashlib.sha256()
with tmp.open('wb') as out:
 for i in range(8):
  with (root/'local_model_transfer'/('part%02d'%i)).open('rb') as src:
   for b in iter(lambda:src.read(8*1024*1024),b''):
    h.update(b);out.write(b)
assert h.hexdigest()=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af',h.hexdigest()
assert not dest.exists(),'Refuse to overwrite an existing checkpoint'
os.replace(tmp,dest)
print(json.dumps({'status':'MODEL_VERIFIED','path':str(dest),'bytes':dest.stat().st_size,'sha256':h.hexdigest()}))
'''
    inp, out, err = c.exec_command('/root/miniconda3/bin/python -', timeout=120)
    inp.write(code)
    inp.channel.shutdown_write()
    result = out.read().decode()
    errors = err.read().decode()
    assert out.channel.recv_exit_status() == 0, errors
    print(result, flush=True)
    PROGRESS.write_text(result)
    c.close()


if __name__ == '__main__':
    main()
