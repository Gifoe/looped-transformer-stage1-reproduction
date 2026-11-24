"""Apply the repository's original zero-shot archive validation and dispatch."""
from pathlib import Path
import runpy
import sys

path = Path(sys.argv[sys.argv.index('--stage2')+1])
sys.path.insert(0,str(path/'code'))
import stage2_score

stage2_score.APPLE_BENCHMARKS=dict.fromkeys(stage2_score.APPLE_BENCHMARKS,0)
runpy.run_path(str(Path(__file__).with_name('mixed_evaluate.py')),run_name='__main__')
