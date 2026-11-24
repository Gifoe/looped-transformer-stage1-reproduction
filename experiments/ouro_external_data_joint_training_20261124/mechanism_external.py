"""Compare same-question C/D in finite writeback and actual generated prefixes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from controller_io import controller
from advantages import token_margin
from evaluate_core import math_diagnostics
from evaluate_generation import generate, summarize
from finite_replay import measurements
from generation import pack_generation
import minimal
from prefix_causality import MARKER, split_tokens, stopping_odds
from task_data import encode_target, hash_order, identity, load, save


def fixed_rows(path, size, tag):
    return sorted(load(path),key=lambda row:hash_order(row['sample_id'], tag))[:size]


def uncertainty(values, seed=261124137):
    v=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    positions=rng.integers(0,len(v),size=(3000,len(v)))
    return dict(n=len(v),mean=float(v.mean()),ci95=np.quantile(v[positions].mean(axis=1),(.025,.975)).tolist())


def finite_summary(rows):
    grouped=defaultdict(dict)
    for row in rows:
        key=(row['view'],row['group'],row['depth'],row['dose'])
        grouped[key].setdefault(row['id'],{})[row['arm']]=row
    output=[]
    for (view, group,depth,dose), by_question in sorted(grouped.items()):
        effects, errors, pair_gap={}, {}, []
        for arm in ('C','D'):
            effects[arm]=uncertainty([item[arm]['q_edited']-item[arm]['q_native'] for item in by_question.values()])
            errors[arm]=float(np.mean([abs(item[arm]['q_edited']-item[arm]['q_native']-item[arm]['linear']) for item in by_question.values()]))
        for item in by_question.values():
            pair_gap.append(item['D']['q_edited']-item['C']['q_edited'])
        output.append(dict(view=view,group=group,depth=depth,dose=dose,paired_D_minus_C=uncertainty(pair_gap),
                           native_relative=effects, mean_absolute_linear_error=errors))
    return output


@torch.inference_mode()
def prefix_replays(archive, model, tok, arms, data, depth, folder):
    result, cost={},{}
    for name,net in arms.items():
        rows,timing=generate(archive,model,tok,net,data,'gsm_process',depth,24,192)
        math_diagnostics(rows)
        result[name],cost[name]=rows,timing
        save(folder/f'gsm8k__{name}__k{depth}.jsonl',rows)
    rows=[]
    marker=tok.encode('\n####',add_special_tokens=False)
    for ix,item in enumerate(data):
        prompt,whole,n=encode_target(tok,item)
        reference=whole[len(prompt):len(prompt)+n-1]
        prefixes={'gold':reference}
        prefixes.update({name:split_tokens(outputs[ix],tok.eos_token_id) for name,outputs in result.items()})
        for position in (0,4,8,16):
            if not all(len(tokens)>=position and not MARKER.search(tok.decode(tokens[:position])) for tokens in prefixes.values()):
                continue
            label=tok.encode(' '+item['references'][0],add_special_tokens=False)[0]
            rows.extend(dict(id=item['sample_id'],depth=depth,origin=name,position=position,prompt=prompt,prefix=value[:position],
                             expected_numeric_token=label)
                        for name,value in prefixes.items())
    device=next(model.parameters()).device
    for arm,net in arms.items():
        points=stopping_odds(model,net,tok,[x['prompt'] for x in rows],[x['prefix'] for x in rows],depth)
        numeric=[]
        for i in range(0,len(rows),16):
            block=rows[i:i+16]
            packed,last,_=pack_generation([x['prompt'] for x in block],[x['prefix']+marker for x in block],
                                           list(range(len(block))),0,device)
            hidden,_=minimal.forward(model,packed,depth,net)
            logits=model.lm_head(hidden[last]).float()
            target=torch.tensor([x['expected_numeric_token'] for x in block],device=device)
            numeric.extend(token_margin(logits,target,tok.eos_token_id).tolist())
        for row,(odds,pred),margin in zip(rows,points,numeric,strict=True):
            row[arm]=dict(eos_log_odds=odds,eos_top1=pred,numeric_start_margin_after_counterfactual_marker=margin)
    for row in rows:
        del row['prompt']
    save(folder/f'prefix_replay__k{depth}.jsonl',rows)
    return dict(arms={name:summarize(records) for name,records in result.items()},cost=cost,prefixes=len(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','previous','model','data','short','explained','out'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    a=parser.parse_args()
    begin=time.monotonic()
    archive,model,tok,hardware=minimal.runtime(a.archive,a.previous,a.model)
    arms={name:controller(path,next(model.parameters()).device) for name,path in (('C',a.short),('D',a.explained))}
    science=fixed_rows(a.data/'scienceqa_development.jsonl',24,'external/mechanism/science')
    math=[row['views']['process'] for row in fixed_rows(a.data/'gsm8k_development.jsonl',24,'external/mechanism/gsm')]
    cases={'science_candidate':[r['views']['short'] for r in science],'science_short':[r['views']['short'] for r in science],
           'science_explanation':[r['views']['explanation'] for r in science],'math_process':math}
    a.out.mkdir(parents=True,exist_ok=True)
    finite=[]
    trajectories={}
    depths=(2,3,4,5,6)
    for k in depths:
        for view,rows in cases.items():
            mode='candidate' if view.endswith('candidate') else 'text'
            measured=list(measurements(archive,model,tok,rows,mode,k,k-1,arms,4))
            for row in measured:
                row['view']=view
            save(a.out/f'finite__{view}__k{k}.jsonl',measured)
            finite.extend(measured)
        if k in (2,4,6):
            trajectories[str(k)]=prefix_replays(archive,model,tok,dict(native=None,**arms),math,k,a.out)
        print('MECHANISM',k,len(finite),flush=True)
    minimal.dump(a.out/'SUMMARY.json',dict(finite=finite_summary(finite),self_prefix=trajectories,
                 model_sha256=hardware['model_sha256'],checkpoint_sha256={'C':minimal.sha(a.short),'D':minimal.sha(a.explained)},
                 duration=time.monotonic()-begin,comparison='matched same-seed epoch 3, 24 fixed development questions per source'))
    print('DONE',len(finite),flush=True)


if __name__=='__main__':
    main()
