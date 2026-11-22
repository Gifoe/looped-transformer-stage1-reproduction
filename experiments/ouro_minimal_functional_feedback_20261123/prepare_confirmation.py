"""Record generation controls and sufficiency criteria while Test is still sealed."""
import argparse
import json
from pathlib import Path

from minimal import dump, sha


def run(out):
    lock=out/'FREEZE.json'
    cfg=json.loads(lock.read_text())
    if 'generation_arms' in cfg:
        return
    validation=json.loads((out/'VALIDATION.json').read_text())
    kl=next(name for name in validation if name.endswith('_t2kl'))
    same=kl.removesuffix('kl')
    for name in (same,kl):
        r=validation[name]
        cfg['arms'][name]=dict(file=name+'.pt',sha256=sha(out/(name+'.pt')),dose=r['dose'],
                              params=r['fit']['total_parameters'],validation=r['counts'])
    cfg['generation_arms']=list(dict.fromkeys([cfg['best'],cfg['generation'],same,kl]))
    cfg['kl_pair']=[same,kl]
    cfg['sufficiency']=dict(retention=.80,parameter_reduction=.70,generation_noninferiority_pp=5.,
       definitions='Pooled signed gains over Native (K2–K4, two metrics) at least 80% separately on ARC and OpenBookQA. '
       'Level A requires all six K2–K4 cells on both datasets at or above historical Feedback-only, <=32768 parameters, '
       'plus generation maintenance. Level B requires >=70% compression, retention on both, a positive gain over prior '
       'at a depth/metric, and no more than five pp pooled degradation in validity or accuracy K2–K4 on either generation set.',
       second_stage='A or B for a single validation-selected method. Zero-shot and paper few-shot comparisons remain independent.')
    dump(lock,cfg)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out)
