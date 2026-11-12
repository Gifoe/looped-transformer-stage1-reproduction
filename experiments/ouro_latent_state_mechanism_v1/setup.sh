#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON_BIN:-python3}"
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Torch installation must match the machine's NVIDIA driver/CUDA support.
# If an installed CUDA torch is usable, reuse it with:
#   PYTHON_BIN=/path/to/python ... (Astra may adapt this script to the server).
python -m pip install -r requirements.txt
python - <<'PY'
import torch, transformers, lm_eval
print('torch:',torch.__version__, 'cuda:',torch.version.cuda,'cuda_available:',torch.cuda.is_available())
print('transformers:',transformers.__version__)
if not torch.cuda.is_available():
    raise SystemExit('No CUDA Torch. Install a compatible CUDA torch wheel into .venv and rerun the validation.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('CUDA device does not report BF16 support; do not silently switch model dtype.')
PY
