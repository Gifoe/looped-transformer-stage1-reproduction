"""Shared path, identity, and inference setup for the dedicated Linux task folder."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure(device='cuda:0'):
    for name in ['tmp', 'cache/hf', 'cache/xdg', 'logs', 'fresh', 'data']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(ROOT / 'tmp'), HF_HOME=str(ROOT / 'cache/hf'),
        XDG_CACHE_HOME=str(ROOT / 'cache/xdg'), PYTHONPYCACHEPREFIX=str(ROOT / 'cache/pycache'),
        TORCH_HOME=str(ROOT / 'cache/torch'), TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'cache/torchinductor'),
        TRITON_CACHE_DIR=str(ROOT / 'cache/triton'), CUDA_CACHE_PATH=str(ROOT / 'cache/cuda'),
        NL_DEVICE=device, NL_QUILL=str(ROOT / 'models/Quill-Qwen3-4B'),
        NL_EMBER=str(ROOT / 'models/Ember-AFM-4.5B'), TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_DISABLE_XET='1')
    os.environ.pop('NL_TINY', None)
    sys.path.insert(0, str(ROOT / 'upstream/pairing'))


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    tmp.replace(path)


def load_frozen():
    configure()
    import torch
    from evaluate import load_pair
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert json.loads((ROOT / 'logs/TINY_TESTS_EXIT.json').read_text())['exit_code'] == 0
    assert json.loads((ROOT / 'fresh/pilot_01/receipt.json').read_text())['status'] == 'passed'
    torch.manual_seed(2021)
    torch.set_num_threads(4)
    pair, toks = load_pair(str(ROOT / 'models/Pairing-final-nl33w'))
    for model in [pair.bridge, *pair.models.values()]:
        model.eval().requires_grad_(False)
    assert not any(p.requires_grad for m in [pair.bridge, *pair.models.values()] for p in m.parameters())
    return pair, toks
