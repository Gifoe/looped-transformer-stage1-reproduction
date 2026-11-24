"""Run the pinned Stage2 zero-shot and MMLU-Pro scorers for the historical T2 only."""
import argparse
import json
from pathlib import Path
import sys

import evaluate_benchmarks as evaluator
import minimal

p=argparse.ArgumentParser(description=__doc__)
for name in ('archive','previous','model','minimal','stage2','unified','followup','historical','out','freeze'):
    p.add_argument(f'--{name}', type=Path, required=True)
p.add_argument('--arm', choices=('t2',), required=True)
p.add_argument('--task', required=True)
a=p.parse_args()
lock=json.loads(a.freeze.read_text())
if lock['entry_sha256']!=minimal.sha(Path(__file__)) or lock['arms']['t2']['sha256']!='117f5f2fe9c5afe363ba8c032f42cd25155f7a88cb465e4e79bfb3bf496c8032':
    raise ValueError('Different T2 evaluation entry or weights')
sys.path[:0]=[str(a.unified),str(a.stage2/'code')]
import seven_unified
import stage2_score

if minimal.sha(Path(seven_unified.__file__))!=lock['stage2_adapter_sha256']:
    raise ValueError('Stage2 archive adapter changed')
stage2_score.APPLE_BENCHMARKS=dict.fromkeys(stage2_score.APPLE_BENCHMARKS,0)
seven_unified.TASKS={name:(0,n,metrics) for name,(_,n,metrics) in seven_unified.TASKS.items()}
evaluator.run(a)
