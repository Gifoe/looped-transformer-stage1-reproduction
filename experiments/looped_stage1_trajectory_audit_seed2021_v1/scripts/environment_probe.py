import importlib.metadata as meta
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
def version(name):
    try:
        return meta.version(name)
    except meta.PackageNotFoundError:
        return None

def fetch(url):
    import requests
    started = time.monotonic()
    try:
        s = requests.Session()
        s.trust_env = False
        r = s.get(url, timeout=(8, 18))
        return dict(url=url, status=r.status_code, elapsed_s=time.monotonic()-started,
                    text=r.text[:1600])
    except Exception as e:
        return dict(url=url, error=repr(e), elapsed_s=time.monotonic()-started)

report = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
              packages={p:version(p) for p in ['torch','transformers','lm-eval','datasets','huggingface-hub','numpy','accelerate','requests']},
              disk={d:shutil.disk_usage(d)._asdict() for d in ['C:/','D:/','E:/']},
              paths={k:os.getenv(k) for k in ['HF_HOME','HUGGINGFACE_HUB_CACHE','HF_ENDPOINT']})
dirs = ['D:/conditional_latent_loop_seed2021_v1','D:/baseline_reproduction_seed2021_v1',
        'C:/Users/fyl412/.cache/huggingface/hub','D:/huggingface','D:/hf_cache','E:/hf_cache']
report['relevant_directories'] = {}
for directory in dirs:
    p = Path(directory)
    if p.exists():
        report['relevant_directories'][directory] = [x.name for x in p.iterdir()][:80]
print(json.dumps(report, indent=2), flush=True)
urls = ['https://api.github.com/repos/L-z-Chen/Training-Free-Looped-Transformer',
        'https://huggingface.co/api/models/ByteDance/Ouro-1.4B',
        'https://huggingface.co/api/models/Qwen/Qwen3-4B-Base',
        'https://hf-mirror.com/api/models/ByteDance/Ouro-1.4B',
        'https://pypi.org/pypi/transformers/json']
with ThreadPoolExecutor(max_workers=5) as pool:
    report['network'] = list(pool.map(fetch, urls))
print(json.dumps(report['network'], indent=2), flush=True)
(ROOT/'logs/environment_probe.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
env = os.environ.copy()
env['PYTHONUTF8']='1'
torch_code = "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('bf16',torch.cuda.is_bf16_supported()); x=torch.ones((4,4),device='cuda',dtype=torch.bfloat16); print(x@x)"
r = subprocess.run([sys.executable, '-u', '-c', torch_code], env=env, capture_output=True, text=True, timeout=90)
report['torch_check'] = dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
(ROOT/'logs/environment_probe.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
print(json.dumps(report['torch_check']), flush=True)
