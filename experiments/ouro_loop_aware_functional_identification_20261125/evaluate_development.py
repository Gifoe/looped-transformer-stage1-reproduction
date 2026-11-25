"""Select among matched controllers using sealed development populations only."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys

import pyarrow.parquet as pq
import torch

from controller_io import controller
from evaluate_generation import candidates,generate
import minimal
from task_data import hash_order,load,save


PANEL={('arc_challenge','candidate'):299,('obqa','candidate'):360,('csqa','candidate'):360,
       ('arc_challenge','content'):192,('csqa','content'):192,('squad','span'):192,('gsm8k','gsm_process'):96}
KEYS={'candidate':'acc_norm','content':'correct','span':'correct','gsm_process':'strict','mmlu':'acc'}


def raw_mmlu(folder):
    result=[]
    labels=list('ABCD')
    for file in sorted(folder.glob('*/dev-*.parquet')):
        topic=file.parent.name
        for index,row in enumerate(pq.read_table(file).to_pylist()):
            id=f'dev_mmlu_{topic}:{index}'
            options='\n'.join(f'{l}. {o}' for l,o in zip(labels,row['choices'],strict=True))
            prompt=f'The following are multiple choice questions (with answers) about {topic.replace("_"," ")}.\n\n{row["question"]}\n{options}\nAnswer:'
            result.append(dict(id=id,task=f'mmlu_{topic}',benchmark='mmlu',doc_id=index,target=int(row['answer']),choices=labels,
                               requests=[dict(type='loglikelihood',args=[prompt,' '+label],idx=i) for i,label in enumerate(labels)]))
    if len(result)!=285 or len({r['id'] for r in result})!=len(result):
        raise ValueError('Expected all 285 official MMLU auxiliary development examples')
    return result


def register(a):
    contents={}
    for (dataset,mode),size in PANEL.items():
        p=a.panel/f'dev_{dataset}.jsonl'
        source=load(p)
        ordered=sorted(source,key=lambda r:hash_order(r['sample_id'],f'loop-aware/dev/{dataset}'))[:size]
        if len(ordered)!=size:
            raise ValueError(f'Development source lacks {dataset}/{mode}')
        contents[(dataset,mode)]=ordered
    contents[('mmlu','mmlu')]=raw_mmlu(a.mmlu)
    return contents


def structured(rows,metric):
    return dict(n=len(rows),value=statistics.mean(r['metrics'][metric] for r in rows))


def matching_protocols(paths, manifest):
    fields=('cohorts','model_sha256','depth','accuracy_rank')
    out=[]
    for folder in paths:
        path=folder/'DEV_PROTOCOL.json'
        previous=json.loads(path.read_text(encoding='utf-8'))
        if any(previous[key]!=manifest[key] for key in fields):
            raise ValueError(f'Development reuse protocol differs at {path}')
        out.append((folder,previous,minimal.sha(path)))
    return out


def reused_receipt(sources, name, checkpoint, stamp, raw, ids, metric):
    for folder, protocol, lock in sources:
        slots=protocol['checkpoint_sha256']
        if name not in slots or slots[name]!=checkpoint:
            continue
        receipt=folder/stamp.name
        record=folder/raw.name
        if not receipt.is_file():
            continue
        doc=json.loads(receipt.read_text(encoding='utf-8'))
        if doc['protocol_sha256']!=lock or doc['sha256']!=minimal.sha(record):
            raise ValueError(f'Development reuse receipt differs at {receipt}')
        rows=load(record)
        if [r['id'] for r in rows]!=ids or structured(rows,metric)!={key:doc[key] for key in ('n','value')}:
            raise ValueError(f'Development reuse population or score differs at {receipt}')
        shutil.copyfile(record,raw)
        return dict(doc, reused_from=str(receipt), source_protocol_sha256=lock)
    return None


@torch.inference_mode()
def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    official=a.out/'SELECTION.json'
    if official.exists():
        raise ValueError('This development selection is already sealed')
    if a.prefill_candidates:
        if not (a.out/'DEV_PROTOCOL.json').is_file():
            raise ValueError('Candidate prefill requires the active study protocol')
        torch.cuda.set_per_process_memory_fraction(.35)
    panel=register(a)
    arch,model,tokenizer,runtime=minimal.runtime(a.archive,a.previous,a.model)
    sys.path[:0]=[str(a.unified),str(a.stage2/'code')]
    from evaluate_benchmarks import candidate_row
    import stage2_score as implementation
    import seven_unified as scorer
    config=json.loads((a.stage2/'scores/mmlu/LOCK.json').read_text())
    encoder=implementation.HarnessEncoder(tokenizer,config['max_length'])
    weights={name:Path(path) for name,path in (item.split('=',1) for item in a.weight)}
    if 't1' not in weights or 'mixed' not in weights:
        raise ValueError('The historical T1 and Mixed development references are required')
    arms={'native':None,**weights}
    hashes={name:minimal.sha(path) if path else None for name,path in arms.items()}
    modes={f'{dataset}/{mode}':dict(ids=[r.get('sample_id',r.get('id')) for r in rows],primary=KEYS[mode])
           for (dataset,mode),rows in panel.items()}
    manifest=dict(cohorts=modes,model_sha256=runtime['model_sha256'],checkpoint_sha256=hashes,depth=4,
                  accuracy_rank=['fewest strong-reference loss endpoints > 2 pp','worst relative gain versus T1','class-balanced gain'])
    locked=a.out/'DEV_PROTOCOL.json'
    if locked.exists() and json.loads(locked.read_text())!=manifest:
        raise ValueError('Development comparison protocol changed')
    minimal.dump(locked,manifest)
    sources=matching_protocols(a.reuse,manifest)
    results={}
    for name,path in arms.items():
        if a.prefill_candidates and name not in a.prefill_candidates:
            continue
        net=controller(path,next(model.parameters()).device)
        results[name]={}
        for (ds,mode),rows in panel.items():
            if a.prefill_candidates and mode not in ('candidate','mmlu'):
                continue
            key=f'{ds}/{mode}'
            stamp=a.out/f'{ds}__{mode}__{name}.json'
            raw=stamp.with_suffix('.jsonl')
            if stamp.exists():
                doc=json.loads(stamp.read_text())
                if doc['sha256']!=minimal.sha(raw) or doc['protocol_sha256']!=minimal.sha(locked):
                    raise ValueError(f'Benchmark result differs at {stamp}')
                results[name][key]=doc
                continue
            previous=reused_receipt(sources,name,hashes[name],stamp,raw,modes[key]['ids'],KEYS[mode])
            if previous is not None:
                previous['protocol_sha256']=minimal.sha(locked)
                minimal.dump(stamp,previous)
                results[name][key]=previous
                print('REUSED DEV',name,key,previous['n'],round(previous['value'],4),flush=True)
                continue
            if mode=='mmlu':
                scored,cost=scorer.shard_score(implementation,minimal,model,encoder,net,rows,config,next(model.parameters()).device)
                observed=[candidate_row(r,scorer.UNIFIED) for r in scored]
            elif mode=='candidate':
                observed,cost=candidates(arch,model,tokenizer,net,rows,4,12)
            else:
                cap,batch={'gsm_process':(192,24),'content':(24,48),'span':(24,32)}[mode]
                observed,cost=generate(arch,model,tokenizer,net,rows,mode,4,batch,cap)
            if [r['id'] for r in observed]!=modes[key]['ids']:
                raise ValueError(f'Development IDs differ at {stamp}')
            if a.prefill_candidates:
                cost=dict(cost,concurrent_development=True)
            save(raw,observed)
            current=dict(**structured(observed,KEYS[mode]),cost=cost,sha256=minimal.sha(raw),protocol_sha256=minimal.sha(locked))
            minimal.dump(stamp,current)
            results[name][key]=current
            print('DEV',name,key,current['n'],round(current['value'],4),flush=True)
    if a.prefill_candidates:
        return
    summary={}
    for name,run in results.items():
        gap={key:record['value']-results['t1'][key]['value'] for key,record in run.items()}
        score_groups=[[v for k,v in gap.items() if k.endswith(('/candidate','/mmlu'))],
                      [v for k,v in gap.items() if k.endswith(('/content','/span'))],
                      [v for k,v in gap.items() if k.endswith('/gsm_process')]]
        summary[name]=dict(gap=gap,native={key:r['value']-results['native'][key]['value'] for key,r in run.items()},
                           score=[-sum(d<-.02 for d in gap.values()),min(gap.values()),statistics.mean(statistics.mean(g) for g in score_groups)])
    labels=[n for n in weights if n not in ('t1','mixed','t2') and not n.startswith('source_')]
    if not labels:
        raise ValueError('No newly trained eligible models')
    best=max(labels,key=lambda k:summary[k]['score'])
    minimal.dump(official,dict(best=best,path=str(weights[best]),checkpoint_sha256=hashes[best],summary=summary,
                               development_protocol_sha256=minimal.sha(locked),all_runs=results))
    print('FROZEN',best,summary[best]['score'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for item in ('archive','previous','model','panel','stage2','mmlu','unified','out'):
        p.add_argument(f'--{item}',type=Path,required=True)
    p.add_argument('--weight',action='append',required=True)
    p.add_argument('--reuse',action='append',type=Path,default=[])
    p.add_argument('--prefill-candidates',nargs='+')
    run(p.parse_args())
