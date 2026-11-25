"""Measure frozen source directions on disjoint question or context groups."""

import argparse
from collections import defaultdict
from pathlib import Path
import json
import time

import torch

from collect_identification import batches, native, pack, readout, source_rows
import minimal
from task_data import hash_order, load, save


DOSES=(.1,.5,1.)
MEASUREMENT_PROTOCOL='ffef5340fe66072e69b251e85bda3be56d193747f52fade21960391b230d9426'
RECEIVER_PROTOCOL='506ece449a98de4d64bc726a170d5c83939c705253b84ccb8573937b370a170f'


def independent_rows(a,name,mode):
    path=a.data/f'{name}.jsonl'
    rows=[r for r in source_rows(path,name,mode)
          if int(hash_order(r.get('fold_key',r['sample_id']),'loop-identification/fold'),16)%4==3]
    rows.sort(key=lambda r:hash_order(r['sample_id'],f'loop-identification/{name}/forecast-confirm'))
    return rows[:a.n]


def edit_with(vector, state, seq, pos, dose):
    u,w,amplitude=vector
    selected=state[seq,pos]
    proposed=amplitude*(selected@w)[:,None]*u
    scaled=proposed*(.45*minimal.rms(selected)/minimal.rms(proposed)).clamp_max(1.)
    return torch.zeros_like(state).index_put((seq,pos),scaled*dose)


@torch.inference_mode()
def measure(archive,model,tok,rows,mode,depth,controls):
    for segment in batches(rows,0,6,2600,mode):
        batch=pack(archive,tok,segment,mode)
        d0,l0,_=native(model,batch,depth)
        p0,a0=readout(model,d0,batch,tok.eos_token_id)
        q0,b0=readout(model,l0,batch,tok.eos_token_id)
        for key,value in controls.items():
            for dose in DOSES:
                p,l,_=native(model,batch,depth,lambda h,s,t:edit_with(value,h,s,t,dose))
                pq,pa=readout(model,p,batch,tok.eos_token_id)
                lq,la=readout(model,l,batch,tok.eos_token_id)
                for i,id in enumerate(batch.ids):
                    d,l=float(pq[i]-p0[i]),float(lq[i]-q0[i])
                    yield dict(id=id,direction=key,dose=dose,direct_gain=d,loop_gain=l,interaction=l-d,
                               native_direct=float(a0[i]),native_loop=float(b0[i]),edited_direct=float(pa[i]),edited_loop=float(la[i]))


def cache_input(path):
    plan=path/'FORECAST.json'
    lock=json.loads(plan.read_text())
    directions=path/'DIRECTIONS.pt'
    if minimal.sha(directions)!=lock['directions_sha256']:
        raise ValueError(f'Changed archived directions in {path}')
    vectors=torch.load(directions,map_location='cpu',weights_only=True)
    return path, minimal.sha(plan), vectors


def archived_controls(inputs, name, depth, fixed, controls):
    accepted={}
    parents={}
    for root, lock, old in inputs:
        path=root/'confirmation'/f'{name}_k{depth}.jsonl'
        stamp=path.with_suffix('.json')
        if not stamp.is_file():
            continue
        receipt=json.loads(stamp.read_text())
        prior=receipt['plan']
        if prior['forecast_sha256']!=lock or receipt['predictions_sha256']!=minimal.sha(path):
            raise ValueError(f'Changed archived replay at {path}')
        version=prior.get('measurement_protocol_sha256',prior['evaluation_code_sha256'])
        if version!=MEASUREMENT_PROTOCOL:
            raise ValueError(f'Unexpected archived measurement implementation at {path}')
        fields=('target','depth','mode','ids','model_sha256','source_sha256','doses')
        if any(prior[field]!=fixed[field] for field in fields):
            continue
        wanted={key for key in fixed['controls'] if key in prior['controls'] and key not in accepted and key in old
                and controls[key]['amplitude']==old[key]['amplitude']
                and all(torch.equal(controls[key][v],old[key][v]) for v in ('u','w'))}
        if not wanted:
            continue
        by=defaultdict(list)
        for row in load(path):
            if row['direction'] in wanted:
                by[row['direction']].append(row)
        for key in sorted(wanted):
            if any([r['id'] for r in by[key] if r['dose']==dose]!=fixed['ids'] for dose in DOSES):
                raise ValueError(f'Invalid reused predictor identities at {path} ({key})')
            accepted[key]=by[key]
            parents[key]=receipt['predictions_sha256']
    return accepted,parents


def run(a):
    old,model,tok,runtime=minimal.runtime(a.archive,a.previous,a.model)
    device=next(model.parameters()).device
    plan=a.forecast/'FORECAST.json'
    lock=json.loads(plan.read_text())
    direction=a.forecast/'DIRECTIONS.pt'
    if minimal.sha(direction)!=lock['directions_sha256']:
        raise ValueError('Frozen directions changed')
    weights=torch.load(direction,map_location='cpu',weights_only=True)
    if a.reuse and minimal.sha(Path(__file__).with_name('collect_identification.py'))!=RECEIVER_PROTOCOL:
        raise ValueError('Cannot reuse archived outcomes after a receiver implementation change')
    cached=[cache_input(path) for path in a.reuse]
    vectors={name:(row['u'].to(device),row['w'].to(device),row['amplitude']) for name,row in weights.items()}
    a.out.mkdir(parents=True,exist_ok=True)
    targets=a.targets or lock['source_order']
    for name in targets:
        mode=lock['sources'][name]['mode']
        source=a.data/f'{name}.jsonl'
        if minimal.sha(source)!=lock['sources'][name]['eligible_inputs_sha256']:
            raise ValueError('Frozen held-out source has changed')
        available={k:v for k,v in vectors.items() if not k.startswith(name+'__')}
        rows=independent_rows(a,name,mode)
        ids=[row['sample_id'] for row in rows]
        if len(ids)!=a.n:
            raise ValueError(f'Insufficient disjoint held-out questions for {name}')
        for k in a.depths:
            out=a.out/f'{name}_k{k}.jsonl'
            receipt=out.with_suffix('.json')
            fixed=dict(target=name,depth=k,mode=mode,ids=ids,forecast_sha256=minimal.sha(plan),model_sha256=runtime['model_sha256'],
                       evaluation_code_sha256=minimal.sha(__file__),source_sha256=minimal.sha(source),doses=list(DOSES),controls=list(available))
            fixed['measurement_protocol_sha256']=MEASUREMENT_PROTOCOL
            if receipt.exists():
                recorded=json.loads(receipt.read_text())
                old_plan=dict(recorded['plan'])
                old_version=old_plan.get('measurement_protocol_sha256',old_plan['evaluation_code_sha256'])
                if old_version==MEASUREMENT_PROTOCOL:
                    old_plan['measurement_protocol_sha256']=MEASUREMENT_PROTOCOL
                    old_plan['evaluation_code_sha256']=fixed['evaluation_code_sha256']
                if old_plan!=fixed or recorded['predictions_sha256']!=minimal.sha(out):
                    raise ValueError(f'Changed held-out results: {out}')
                print('REUSED',name,k,flush=True)
                continue
            started=time.perf_counter()
            reused,parents=archived_controls(cached,name,k,fixed,weights)
            active={key:value for key,value in available.items() if key not in reused}
            records=[row for group in reused.values() for row in group]
            if active:
                records.extend(measure(old,model,tok,rows,mode,k,active))
            if len(records)!=len(ids)*len(available)*len(DOSES):
                raise ValueError('Incomplete held-out interventions')
            torch.cuda.synchronize()
            save(out,records)
            minimal.dump(receipt,dict(plan=fixed,predictions_sha256=minimal.sha(out),seconds=time.perf_counter()-started,
                                      reused_direction_receipts=parents,new_directions=len(active)))
            print('DONE',name,k,len(ids),len(records),'reused',len(reused),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('archive','previous','model','data','forecast','out'):
        p.add_argument(f'--{field}',type=Path,required=True)
    p.add_argument('--n',default=48,type=int)
    p.add_argument('--targets',nargs='+')
    p.add_argument('--depths',nargs='+',type=int,default=(4,3,5))
    p.add_argument('--reuse',action='append',type=Path,default=[])
    run(p.parse_args())
