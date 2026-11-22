"""Score the locked readout operators with live frozen Ouro and isolated timing."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch

import calibration
from bridge import configure, dump, rows
from dual import DEPTHS, base_scores, features_for_gate, gate_engine, split_scores, token_dose, weighted_dose

BATCH_QUESTIONS=4


def inputs(old,tok,records):
    candidates,meta=old.encode(tok,records)
    per=[[] for _ in records]
    for item in candidates:
        per[item[2]].append(item)
    ordered=sorted(range(len(records)),key=lambda q:max(len(x[0]) for x in per[q]))
    return meta, [[i for q in ordered[j:j+BATCH_QUESTIONS] for i in per[q]] for j in range(0,len(ordered),BATCH_QUESTIONS)]


@torch.no_grad()
def copy_diagnostics(meta,items,after,native=None):
    for j,(_,_,q,a) in enumerate(items):
        meta[q]['scores'].setdefault('candidate',[None]*len(meta[q]['labels']))[a]=float(after[j])
        if native is not None:
            meta[q]['scores'].setdefault('native',[None]*len(meta[q]['labels']))[a]=float(native[j])


@torch.no_grad()
def pass_method(old,af,amp,model,net,meta,batches,k,dose,gconf,retain_native):
    def run(items):
        current,group,target=af.forward(model,old,items,k)
        orig=base_scores(amp,model,current,k,target,group,len(items)) if gconf or retain_native else None
        out,details=net(current,k,dose)
        fraction=None
        if gconf:
            qids=list(dict.fromkeys(x[2] for x in items))
            lens=[meta[q]['characters'] for q in qids]
            options=[len(x) for x in lens]
            flat=[c for a in lens for c in a]
            f=features_for_gate(amp,current,details,group,flat,options,orig,k)
            engine, threshold=gconf
            chosen=weighted_dose(engine(f),threshold)
            frac=float(chosen.sum())
            gain,shift,_,w,_=details
            out=(current['raw'][:,k-1].float()+dose*token_dose(chosen,options,group)*(gain*w+shift)).to(torch.bfloat16)
            fraction=(len(qids),frac)
        vocab=model.lm_head(amp.normalized(model,out)).float()
        out=old.scores(vocab,group.cpu().numpy(),target,len(items))
        return out,orig.cpu().tolist() if retain_native else None,fraction

    run(batches[0]); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed=[]; total=0; modified=0.
    for attempt in batches:
        start=time.perf_counter()
        values,base,counts=run(attempt)
        elapsed.append(time.perf_counter()-start)
        copy_diagnostics(meta,attempt,values,base)
        if counts:
            total+=counts[0]; modified+=counts[1]
    torch.cuda.synchronize()
    n=sum(len(batch) for batch in batches)
    return dict(wall_s=sum(elapsed),median_batch_ms=float(np.median(elapsed)*1000),peak_bytes=torch.cuda.max_memory_allocated(),
                candidates=n,backbone_batch_calls=len(batches),loops=k,head_calls_per_batch=2 if gconf or retain_native else 1,
                parameter_count=sum(p.numel() for p in net.parameters()),
                gate_parameter_count=0,
                active_fraction=modified/total if total else 1.,batch_questions=BATCH_QUESTIONS,
                padded_token_loops=sum(len(b)*max(len(x[0])-1 for x in b)*k for b in batches))


def merge(table,new,label,k):
    assert len(table)==len(new)
    for a,b in zip(table,new,strict=True):
        assert a['id']==b['id']
        a['scores'][f'{label}_k{k}']=b['scores']['candidate']
        if 'native' in b['scores']:
            a['scores'][f'native_k{k}']=b['scores']['native']


def write(old,path,table):
    part=path.with_suffix('.tmp')
    old.write_predictions(part,table)
    part.replace(path)


def run(model_path,prior,arc,corpus,out,external=None):
    old,af,amp=configure(prior,arc)
    lock=json.loads((out/'READOUT_FREEZE.json').read_text())
    for f,digest in lock['weights'].items():
        assert af.sha(out/f)==digest
    calconf=json.loads((out/'CALIBRATION_FREEZE.json').read_text())['chosen']
    cal=calibration.load(calconf)
    model,tok,runtime=old.load_runtime(model_path)
    model.requires_grad_(False)
    if external:
        data=pq.read_table(external).to_pylist()
        name='OPENBOOKQA'
        assert len(data)==500
        content=external
    else:
        content=corpus/'test.parquet'
        assert af.sha(content)==json.loads((arc/'results/test_input.json').read_text())['sha256']
        data=old.read_data(corpus,'test'); name='ARC'
        assert len(data)==1172
    for part in ('train','validation'):
        seen={x['id'] for x in old.read_data(corpus,part)}
        assert external or seen.isdisjoint(x['id'] for x in data)
    metadata,batches=inputs(old,tok,data)
    combined=[dict(r,scores={}) for r in metadata]
    cost={}; progress=out/f'progress_{name.lower()}'
    progress.mkdir(exist_ok=True)
    cases=('dual','gate','confidence','teacher','shuffled','functional') if not external else ('dual','gate','confidence')
    for arm in cases:
        cfg=lock['config'][arm]; current=amp.operator(out/cfg['file'])
        gconf=(gate_engine(cfg['gate']),cfg['threshold']) if 'gate' in cfg else None
        for k in DEPTHS:
            checkpoint=progress/f'{arm}_k{k}.jsonl'; stat=progress/f'{arm}_k{k}.json'
            if not (checkpoint.exists() and stat.exists()):
                table=[dict(r,scores={}) for r in metadata]
                measure=pass_method(old,af,amp,model,current,table,batches,k,cfg['dose'],gconf,bool(external and arm=='gate'))
                if gconf:
                    measure['gate_parameter_count']=sum(np.asarray(v).size for v in cfg['gate']['state'].values())
                # JSON roundtrip keeps hardware counts plain Python scalars.
                measure['gate_parameter_count']=int(measure['gate_parameter_count'])
                write(old,checkpoint,table); dump(stat,measure)
            merge(combined,rows(checkpoint),arm,k)
            cost[f'{arm}_k{k}']=json.loads(stat.read_text())
            print(name,arm,k,cost[f'{arm}_k{k}']['wall_s'],flush=True)
    reference=combined if external else rows(arc/'results/test_predictions.jsonl')
    assert all(all(a[f]==b[f] for f in ('id','labels','characters','answer_tokens','gold')) for a,b in zip(combined,reference,strict=True))
    calibrated,seconds=calibration.apply(cal,reference)
    for a,b in zip(combined,calibrated,strict=True):
        a['scores'].update(b['scores'])
    cost['scoreonly']=dict(cpu_postprocessing_all_four_s=seconds,parameter_count=calconf['parameter_count'],base_scores='archived' if not external else 'first head of gate')
    write(old,out/f'{name}_READOUT.jsonl',combined)
    dump(out/f'{name}_READOUT_COST.json',cost)
    dump(out/f'{name}_INPUT.json',dict(n=len(data),sha256=af.sha(content),runtime=runtime,
          dual_lock_sha256=af.sha(out/'READOUT_FREEZE.json'),calibration_lock_sha256=af.sha(out/'CALIBRATION_FREEZE.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','previous','archive','corpus','out'):
        p.add_argument(name,type=Path)
    p.add_argument('--external',type=Path)
    a=p.parse_args();run(a.model,a.previous,a.archive,a.corpus,a.out,a.external)
