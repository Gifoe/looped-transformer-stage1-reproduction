"""Run the existing scorer with the original Stage2 zero-shot cohorts."""
import argparse
import json
from pathlib import Path
import sys
import evaluate_benchmarks as evaluator
import minimal

p=argparse.ArgumentParser(description=__doc__)
for name in ('archive','previous','model','minimal','stage2','unified','followup','historical','out','freeze'):
    p.add_argument(f'--{name}',type=Path,required=True)
p.add_argument('--arm',choices=('mixed',),required=True)
p.add_argument('--task',required=True)
args=p.parse_args()
lock=json.loads(args.freeze.read_text())
if lock['entry_sha256']!=minimal.sha(Path(__file__)) or lock['arms']['mixed']['sha256']!='ec2e55a64564f608cfeb245da079015db4347a9391072cba7f2949db05185b87':
    raise ValueError('Different Mixed evaluation entry or weights')
sys.path.insert(0,str(args.unified))
import seven_unified
if minimal.sha(Path(seven_unified.__file__))!=lock['stage2_adapter_sha256']:
    raise ValueError('Stage2 archive adapter changed')
seven_unified.TASKS={name:(0,n,metrics) for name,(_,n,metrics) in seven_unified.TASKS.items()}
evaluator.run(args)
