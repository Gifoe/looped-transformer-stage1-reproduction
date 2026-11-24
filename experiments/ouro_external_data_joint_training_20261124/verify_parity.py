"""Check training masks against the exact incremental generation input path."""

import argparse
from pathlib import Path

from controller_io import controller
import minimal
from task_data import load
from train_generative import check_protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','previous','model','data','checkpoint','out'):
        p.add_argument(f'--{name}',type=Path,required=True)
    a=p.parse_args()
    archive,model,tok,_=minimal.runtime(a.archive,a.previous,a.model)
    net=controller(a.checkpoint,next(model.parameters()).device)
    examples=[]
    for source,tag in (('csqa','short'),('gsm8k','final')):
        rows=load(a.data/f'{source}_train.jsonl')
        examples.append(min((r['views'][tag] for r in rows),key=lambda r:(r['response_tokens'],r['sample_id'])))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    check_protocol(archive,model,tok,net,examples,(2,3,4),a.out)
    print('DONE',flush=True)


if __name__=='__main__':
    main()
