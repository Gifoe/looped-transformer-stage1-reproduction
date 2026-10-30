"""Official CPU tests on the replacement host; captures actual pytest exit."""
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from linux_runtime import ROOT, configure, save

configure('cpu')
import torch

packages = ['torch', 'transformers', 'peft', 'numpy', 'scipy', 'datasets', 'pytest', 'huggingface_hub', 'safetensors', 'accelerate']
save(ROOT / 'logs/linux_environment_manifest.json', dict(python=sys.version, executable=sys.executable,
    platform=platform.platform(), packages={p: importlib.metadata.version(p) for p in packages},
    cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
    capability=torch.cuda.get_device_capability(0), bf16=torch.cuda.is_bf16_supported(),
    scope='CPU implementation tests; no task-performance claim'))
env = dict(os.environ, NL_TINY='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', NP_UPSTREAM=str(ROOT / 'upstream'), NP_MODEL_ROOT=str(ROOT / 'models'))
files = ['tests/test_link.py', 'tests/test_interleaved.py', 'tests/test_frozen.py', 'tests/test_leakage.py', 'tests/test_measurement_audit.py']
start = time.time()
with (ROOT / 'logs/linux_tiny_tests.out').open('xb') as out, (ROOT / 'logs/linux_tiny_tests.err').open('xb') as err:
    code = subprocess.run([sys.executable, '-m', 'pytest', '-q', *files,
        '--junitxml=' + str(ROOT / 'logs/linux_tiny_tests.xml')], cwd=ROOT / 'upstream',
        env=env, stdout=out, stderr=err, timeout=900).returncode
save(ROOT / 'logs/TINY_TESTS_EXIT.json', dict(exit_code=code, files=files, device='cpu',
    upstream_commit='7e7fcc401a8243f94bca94070db0993d64277874', elapsed_seconds=time.time()-start,
    gpu_worker_seconds=0))
if code:
    raise SystemExit(code)
subprocess.run([sys.executable, str(ROOT / 'scripts/check_causal_trace.py')], env=env, cwd=ROOT, check=True, timeout=300)
