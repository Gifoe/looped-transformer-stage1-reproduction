"""CPU-only, deterministic rebuild of all tables and figures from raw records."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
for name in ['analyze.py','analyze_qwen.py','additional_diagnostics.py','loopcd_diagnostics.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,check=True)
print('All raw-result tables and figures rebuilt.',flush=True)
