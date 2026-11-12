"""Use the shared verified stager for CUDA wheels not already staged by Sol."""
import json
from pathlib import Path
import root_stage_cuda_wheels as stage

metadata=json.loads(Path('D:/ouro-e4-e5-20261112/exec/cuda_metadata.json').read_text())
completed={'nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12','nvidia-cuda-cupti-cu12'}
stage.PACKAGES=[]
for package,artifact in metadata.items():
    if package in completed:
        continue
    name=artifact['filename']
    version=name.split('-')[1]
    stage.PACKAGES.append((package,version,name,artifact['size'],artifact['digests']['sha256']))
stage.STATE=Path('D:/ouro-e4-e5-20261112/results/remaining_cuda_staging.json')
stage.main()
