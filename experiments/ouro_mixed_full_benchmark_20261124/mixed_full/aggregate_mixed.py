"""Verify complete Mixed and archived rows, then calculate matched accuracy and costs."""

import argparse
import ast
import copy
import csv
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from evaluate_benchmarks import candidate_row, mmlu_metrics
from evaluate_core import verified
import paired_analysis


NAMES = {'arc_challenge': 'ARC-Challenge', 'arc_easy': 'ARC-Easy', 'sciq': 'SciQ', 'mmlu': 'MMLU',
         'hellaswag': 'HellaSwag', 'winogrande': 'WinoGrande', 'piqa': 'PIQA', 'mmlu_pro': 'MMLU-Pro'}
METHODS = {'native': 'native_k4', 'fixed': 'loopcd_fixed_k4', 'adaptive': 'loopcd_adaptive_k4'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def generation_protocol(current, previous):
    equivalence = {}
    for name in ('prompt_items','extract','batches','cache_type','forward','prepare','greedy'):
        functions=[]
        for path in (current, previous):
            fn = next(f for f in ast.parse(path.read_text()).body if isinstance(f,ast.FunctionDef) and f.name == name)
            if name == 'greedy':
                fn = copy.deepcopy(fn)
                fn.args.kwonlyargs, fn.args.kw_defaults = [], []
                for node in ast.walk(fn):
                    for field,value in ast.iter_fields(node):
                        if isinstance(value,list):
                            setattr(node,field,[n for n in value if not (
                                isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='parity')])
            functions.append(ast.dump(fn))
        if functions[0] != functions[1]:
            raise ValueError(f'MMLU-Pro generation function changed: {name}')
        equivalence[name]=sha256(functions[0].encode()).hexdigest()
    return equivalence


def archived_mmlu(scorer, args, freeze, runtime, load_outputs):
    from transformers import AutoTokenizer
    import generation_followup_run as old

    path=args.followup/'results/mmlu_pro'
    original=path/'LOCK.json'
    reference=args.followup/'generation_followup_run_reference.py'
    main=Path(old.__file__)
    source=args.followup/'data/mmlu_pro.parquet'
    if scorer.sha(original)!=freeze['mmlu_pro_reference_lock_sha256'] or scorer.sha(main)!=freeze['mmlu_pro_code_sha256']:
        raise ValueError('MMLU-Pro frozen code or archive changed')
    lock=read(original)
    if (scorer.sha(reference)!=lock['code_sha256'] or scorer.sha(source)!=lock['dataset']['sha256']
            or scorer.sha(source)!=freeze['mmlu_pro_sha256'] or lock['model']['files_sha256'] != runtime['files_sha256']
            or (lock['depth'],lock['max_new_tokens'],lock['batch_size'],lock['capacity_tokens'],lock['shots'],lock['decoding'])
               !=(4,8,96,14500,0,'greedy')):
        raise ValueError('Original MMLU-Pro settings, source, or tokenizer mismatch')
    equivalence=generation_protocol(main,reference)
    provenance=dict(original_runner_sha256=scorer.sha(reference),active_runner_sha256=scorer.sha(main),
                    equal_greedy_without_parity=equivalence, rollout_lock_sha256=scorer.sha(original),
                    original_protocol=lock)
    if not load_outputs:
        return provenance
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    items=old.prompt_items(source,'mmlu_pro',tokenizer)
    batches=list(old.batches(items,8,96,14500))
    if len(items)!=12032:
        raise ValueError('Wrong MMLU-Pro population')
    outputs, costs={},{}
    for method in METHODS:
        directory=path/method
        records=[]
        seconds, peak = 0., 0
        for i,batch in enumerate(batches):
            raw=directory/f'{i:04d}.jsonl'
            cost=read(raw.with_suffix('.cost.json'))
            chunk,digest=scorer.read_jsonl(raw)
            ids=[item.key for item in batch]
            if cost['sha256']!=digest or cost['ids']!=ids or [r['id'] for r in chunk]!=ids:
                raise ValueError(f'MMLU-Pro historical shard is not paired: {method}/{i}')
            records.extend(chunk)
            seconds += cost['seconds']
            peak = max(peak,cost['peak_bytes'])
        stored,digest=scorer.read_jsonl(path/f'{method}.jsonl')
        if stored != records:
            raise ValueError(f'MMLU-Pro final archive differs from its shards: {method}')
        outputs[method]=mmlu_metrics(records,[item for batch in batches for item in batch],old)
        costs[method]=dict(seconds=seconds,peak_allocated_bytes=peak,sha256=digest)
    return provenance, [item.key for batch in batches for item in batch], outputs, costs


def pct(x):
    return f'{100*x:.2f}%'


def contrast(row):
    lo,hi=row['paired_ci95']
    return f"{100*row['delta']:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]"


def run(a):
    sys.path[:0] = [str(a.unified),str(a.stage2/'code'),str(a.followup)]
    import seven_unified as scorer
    import stage2_score as stage

    stage.APPLE_BENCHMARKS=dict.fromkeys(stage.APPLE_BENCHMARKS,0)
    scorer.TASKS={name:(0,n,metrics) for name,(_,n,metrics) in scorer.TASKS.items()}
    freeze=read(a.out/'FREEZE.json')
    seal=scorer.sha(a.out/'FREEZE.json')
    index=read(a.stage2/'COMPARISON.json')
    reports={}
    records=[]
    markdown=['# Ouro-1.4B Mixed 完整 K4 评测','',
              '固定单一 4,096 参数 Mixed 权重。前七项使用 Stage2 的完整零样本候选评分，MMLU-Pro 使用原始零样本、自回归贪心生成与解析。历史基线仅用于逐题配对。','',
              '| Benchmark | n | 主指标 | Native Ouro | LoopCD Fixed | LoopCD Adaptive | Mixed | 相对 Ouro pp [95%] | 相对最佳 LoopCD pp [95%] | Mixed 秒数 | 峰值显存 GiB |',
              '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |']
    for task, label in NAMES.items():
        mode='generated' if task=='mmlu_pro' else 'candidate'
        path=a.out/'predictions'/f'{task}__mixed__k4__{mode}.jsonl'
        if not path.with_suffix('.json').exists():
            continue
        doc=read(path.with_suffix('.json'))
        if doc['plan']['freeze_sha256']!=seal or doc['plan']['checkpoint_sha256']!=freeze['arms']['mixed']['sha256']:
            raise ValueError(f'Unexpected model or protocol for {task}')
        if task=='mmlu_pro':
            archive, ids, historical, baseline_cost = archived_mmlu(scorer,a,freeze,doc['runtime'],True)
            keys=('correct','parseable','strict_format','strict')
            primary='correct'
        else:
            panel=scorer.check_archive(a,stage,doc['runtime'],task,index)
            ids=[item['id'] for item in panel.requests]
            archive=panel.public
            historical={arm:[candidate_row(row,name) for row in panel.references] for arm,name in METHODS.items()}
            baseline_cost=dict(joint_archived_scoring=archive['archived_joint_scoring_cost'])
            keys=('acc','acc_norm')
            primary=freeze['primary'][task]
        paired_analysis.METRICS.update(keys)
        own,cost=verified(path,doc['plan'],ids)
        if len(own)!=len(set(ids)) or len(ids)!=freeze['expected'][task]:
            raise ValueError(f'Wrong row count for {task}')
        pairs={arm:{key:value for key,value in paired_analysis.compare(rows,own).items() if key in keys}
               for arm,rows in historical.items()}
        best=max(('fixed','adaptive'),key=lambda key:pairs[key][primary]['reference'])
        seconds=cost['seconds'] if mode=='generated' else cost['elapsed_seconds']
        peak=cost['cuda_peak'] if mode=='generated' else cost['peak_allocated_bytes']
        for arm, by_metric in pairs.items():
            for metric,pair in by_metric.items():
                records.append(dict(benchmark=task,n=pair['n'],primary=metric==primary,metric=metric,reference=arm,
                                    mixed=pair['controller'],baseline=pair['reference'],delta=pair['delta'],
                                    ci_low=pair['paired_ci95'][0],ci_high=pair['paired_ci95'][1],rescued=pair['rescued'],
                                    harmed=pair['harmed'],mcnemar=pair['exact_mcnemar'],seconds=seconds,peak_cuda_bytes=peak))
        reports[task]=dict(n=len(ids),metric=primary,source=archive,baselines=baseline_cost,
                           paired=pairs,best_loopcd=best,mixed_cost=cost,rows_sha256=doc['rows_sha256'])
        n=pairs['native'][primary]; f=pairs['fixed'][primary]; d=pairs['adaptive'][primary]
        line=(f"| {label} | {len(ids):,} | {primary} | {pct(n['reference'])} | {pct(f['reference'])} | {pct(d['reference'])} | "
              f"{pct(n['controller'])} | {contrast(n)} | {contrast(pairs[best][primary])} | {seconds:.2f} | {peak/2**30:.2f} |")
        markdown.append(line)
        print(line,flush=True)
    total=sum(r['n'] for r in reports.values())
    if a.final and (len(reports)!=8 or total!=43769):
        raise ValueError(f'Incomplete full evaluation: {len(reports)} tasks, {total} items')
    if not records:
        return
    scorer.json_write(a.out/'RESULTS.json',dict(freeze_sha256=seal,completed_questions=total,
                      completed_tasks=len(reports),results=reports))
    temporary=a.out/'RESULTS.csv.partial'
    with temporary.open('w',encoding='utf-8',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(a.out/'RESULTS.csv')
    markdown.extend(['',f'已验证 {total:,} 道题。区间为逐题配对 95% 区间。最佳 LoopCD 为每项主指标上历史成绩较高的一支。',
                     'Mixed 秒数为全部推理批耗时之和。显存为 CUDA 记录的峰值已分配显存。候选任务的历史耗时是多臂合计记录。MMLU-Pro 单基线耗时与显存保留于 JSON。','',
                     '[逐项统计和成本](RESULTS.json) · [全部指标及救回、退化数](RESULTS.csv) · [锁定配置](FREEZE.json)',''])
    (a.out/'REPORT_ZH.md').write_text('\n'.join(markdown),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('out','minimal','unified','stage2','followup','model'):
        parser.add_argument(f'--{name}',required=True,type=Path)
    parser.add_argument('--final',action='store_true')
    run(parser.parse_args())
