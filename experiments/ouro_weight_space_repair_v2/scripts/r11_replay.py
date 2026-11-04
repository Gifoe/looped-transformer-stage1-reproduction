"""Replay completed numerical evidence without loading the model.

Requires restored inherited inputs and source files. The immutable core rejects
partial files, changed sources, stage locks, signatures or candidate outcomes.
Exact numerical acceptance uses the pinned experiment environment.
"""
import argparse,json
from r11_evaluate import table,final_report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['parity','confirmation','validation','test'],default='parity');a=p.parse_args()
    result=final_report() if a.phase=='test' else table(a.phase)[1]
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
