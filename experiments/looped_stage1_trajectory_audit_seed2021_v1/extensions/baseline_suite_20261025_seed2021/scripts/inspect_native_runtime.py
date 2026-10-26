"""CPU-only runtime provenance; no checkpoint loading or CUDA allocation."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig

root = Path(__file__).resolve().parents[1]
print(json.dumps(dict(phase='stdlib', pid=os.getpid(), python=sys.version,
                     executable=sys.executable)), flush=True)
import numpy as np
print(json.dumps(dict(phase='numpy', version=np.__version__, file=np.__file__)), flush=True)
import torch
print(json.dumps(dict(phase='torch', version=torch.__version__, file=torch.__file__,
                     cuda_build=torch.version.cuda)), flush=True)
torch.set_num_threads(4)
a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
b = a*torch.rsqrt(a.pow(2).mean(-1, keepdim=True)+1e-6)
assert bool(torch.isfinite(b).all())
try:
    from threadpoolctl import threadpool_info
    pools = threadpool_info()
except ImportError:
    pools = 'threadpoolctl not installed'
variables = ['PYTHONMALLOC', 'PYTHON_GIL', 'PYTHONHASHSEED', 'OMP_NUM_THREADS',
             'MKL_THREADING_LAYER', 'KMP_DUPLICATE_LIB_OK', 'CUDA_LAUNCH_BLOCKING',
             'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'USE_TORCH', 'USE_TF',
             'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG']
versions = {}
for name in ['torch', 'numpy', 'transformers', 'tokenizers', 'safetensors', 'scipy', 'pandas']:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
result = dict(inspection_completed=True, pid=os.getpid(), python=sys.version,
    executable=sys.executable, platform=platform.platform(), versions=versions,
    torch_file=torch.__file__, numpy_file=np.__file__, cuda_build=torch.version.cuda,
    gil_enabled=sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else 'unavailable',
    gil_disabled_build=sysconfig.get_config_var('Py_GIL_DISABLED'),
    threadpools=pools, torch_config=torch.__config__.show(),
    relevant_environment={name: os.environ.get(name) for name in variables},
    scope='CPU imports, provenance and small CPU norm calculation only. No model, CUDA allocation or claim that GPU runtime stability is restored.')
(root/'logs/native26_runtime_inspection.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
