"""Verify original questions, response masking, and matched A/B/C/D exposure without GPU."""

import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer

from train_external import epoch_schedule, features, pools


def projection(plan, domains):
    return [(depth,[(r['sample_id'],tag) for r,tag in b if r['dataset'] in domains]) for depth,b in plan]


def check(data, model, config):
    seed=json.loads(config.read_text())['seeds'][0]
    rows=pools(data)
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=True)
    encoded=features(tok,rows)
    counts={recipe:Counter() for recipe in 'ABCD'}
    for epoch in range(1,4):
        schedules={recipe:list(epoch_schedule(rows,recipe,seed,epoch)) for recipe in 'ABCD'}
        for recipe,plan in schedules.items():
            if len(plan)!=500 or len({r['sample_id'] for _,batch in plan for r,_ in batch})!=4000:
                raise ValueError(f'Unbalanced source coverage in {recipe} epoch {epoch}')
            for depth,batch in plan:
                tags=Counter(v for _,v in batch)
                if len(batch)!=8 or tags['process']!=3 or tags['final']!=1 or tags['explanation']!=(recipe=='D'):
                    raise ValueError(f'Wrong views at {recipe} epoch {epoch}')
                counts[recipe].update((depth,r['dataset'],r['views'][v]['task_type']) for r,v in batch)
        if projection(schedules['A'],rows)!=projection(schedules['B'],rows):
            raise ValueError('A and B differ in training order or views')
        shared={'csqa','gsm8k'}
        if any(projection(plan,shared)!=projection(schedules['A'],shared) for plan in schedules.values()):
            raise ValueError('C/D change common-source exposures')
        ids=lambda plan: [[row['sample_id'] for row,_ in batch] for _,batch in plan]
        if ids(schedules['C'])!=ids(schedules['D']):
            raise ValueError('C/D no longer use the same questions')
    doc=dict(seed=seed,episodes={r:{f'k{k}/{ds}/{v}':n for (k,ds,v),n in sorted(c.items())} for r,c in counts.items()},
             encoded_views=len(encoded))
    (data/'SCHEDULE_PREFLIGHT.json').write_text(json.dumps(doc,indent=2)+'\n')
    print('Verified 1,500 updates per configuration and',len(encoded),'tokenized response views.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','model','config'):
        p.add_argument(f'--{name}',type=Path,required=True)
    args=p.parse_args()
    check(args.data,args.model,args.config)
