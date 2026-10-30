"""Inference-only prerequisite for a measured warm-up training smoke test."""
import json
import subprocess
import sys
import time
from linux_runtime import ROOT,configure,save

configure()
out=ROOT/'data/own_replies'
assert not (out/'replies.json').exists(),'preserve existing own replies'
start=time.time();state=dict(status='running',started_unix=start,protocol='official prep.py --n_steer 0 --reply 48 --bs 16',
    purpose='own targets required for warm-up; missing steering targets are not claimed as full-mixture data')
save(ROOT/'logs/OWN_REPLY_STATE.json',state)
try:
    code=subprocess.run([sys.executable,str(ROOT/'upstream/pairing/prep.py'),'--out',str(out),'--n_steer','0','--reply','48','--bs','16'],cwd=ROOT,timeout=3600).returncode
    state.update(status='complete' if code==0 else 'failed',exit_code=code)
    if code:raise SystemExit(code)
finally:
    state.update(finished_unix=time.time(),gpu_worker_seconds=time.time()-start);save(ROOT/'logs/OWN_REPLY_STATE.json',state)
