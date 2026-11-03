"""Create a dedicated environment, sharing existing CUDA packages read-only."""
from pathlib import Path
import hashlib,json,os,subprocess,time

ROOT=Path(__file__).resolve().parents[1]
OLD=Path(os.environ.get('OURO_SHARED_ENV','/home/newuser1/neural_pairing_hidden_state_audit_v1_20261030/env'))
os.environ.update(TMPDIR=str(ROOT/'tmp'),PIP_CACHE_DIR=str(ROOT/'cache/pip'),
    HF_HOME=str(ROOT/'cache/hf'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'),
    PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'),PIP_DISABLE_PIP_VERSION_CHECK='1')
for d in ['logs','data/raw','results','configs','models','figures','tmp','cache/pip','cache/hf','cache/xdg','cache/pycache']:(ROOT/d).mkdir(parents=True,exist_ok=True)
receipt=ROOT/'logs/ENV_SETUP.json';start=time.time()
def save(**data):receipt.write_text(json.dumps(dict(updated_unix=time.time(),elapsed_seconds=time.time()-start,**data),indent=2))
def command(args):
    p=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    with (ROOT/'logs/env_setup.out').open('a') as f:f.write(p.stdout)
    if p.returncode:raise RuntimeError('Environment command failed; see env_setup.out')
try:
    save(status='creating')
    assert OLD.joinpath('bin/python').exists()
    shared=next((OLD/'lib').glob('python*/site-packages'))
    originals={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [shared/'torch/__init__.py',shared/'transformers/__init__.py']}
    env=ROOT/'env';assert not env.exists(),'Preserve existing environment; inspect its receipt'
    command(['python3','-m','venv',str(env)])
    site=next((env/'lib').glob('python*/site-packages'))
    (site/'shared_cuda_readonly.pth').write_text(str(shared)+'\n')
    save(status='installing',shared_readonly_site=str(shared))
    command([str(env/'bin/python'),'-m','pip','install','transformers==4.55.4','tokenizers==0.21.4','huggingface_hub==0.34.4','peft==0.17.1','lm_eval==0.4.9'])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in originals.items()),'Shared base environment changed'
    command([str(env/'bin/python'),'-c','import torch,transformers,lm_eval,peft; from lm_eval.models.huggingface import HFLM; from lm_eval.api.task import ConfigurableTask; print(torch.__version__,transformers.__version__)'])
    save(status='complete',shared_readonly_site=str(shared),base_files_unchanged=True,gpu_worker_seconds=0)
except BaseException as e:
    save(status='failed',error_type=type(e).__name__,error=str(e),gpu_worker_seconds=0);raise
