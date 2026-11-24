"""Pair the full T2 panel with T1, Mixed, and the original Ouro and LoopCD runs."""

import argparse
import csv
from pathlib import Path
import sys

from evaluate_benchmarks import candidate_row
import paired_analysis

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'mixed_full'),str(ROOT/'t1_full')]
from aggregate_mixed import NAMES, METHODS, archived_mmlu, contrast, pct, read
from aggregate_t1 import checked_arm


BASELINES=('native','fixed','adaptive','mixed','t1')


def run(a):
    sys.path[:0]=[str(a.unified),str(a.stage2/'code'),str(a.followup)]
    import seven_unified as scorer
    import stage2_score as scorer_impl

    scorer_impl.APPLE_BENCHMARKS=dict.fromkeys(scorer_impl.APPLE_BENCHMARKS,0)
    scorer.TASKS={name:(0,n,metrics) for name,(_,n,metrics) in scorer.TASKS.items()}
    roots={'mixed':a.mixed,'t1':a.t1,'t2':a.out}
    locks={arm:read(root/'FREEZE.json') for arm,root in roots.items()}
    seals={arm:scorer.sha(root/'FREEZE.json') for arm,root in roots.items()}
    lock=locks['t2']
    for arm in ('mixed','t1'):
        if lock[f'{arm}_protocol_sha256']!=seals[arm]:
            raise ValueError(f'T2 does not reference the published {arm} panel')
        for field in ('depth','expected','model_sha256','stage2_sha256','seven_requests_sha256','mmlu_pro_sha256','mmlu_pro_code_sha256'):
            if lock[field]!=locks[arm][field]:
                raise ValueError(f'T2 and {arm} differ on {field}')
    index=read(a.stage2/'COMPARISON.json')
    results, records={}, []
    lines=['# 历史 Rank-1 T2 完整 K4 评测','',
           '固定 4,096 参数 T2，在与 Mixed、T1 相同的七套全量零样本候选请求上评分。MMLU-Pro 复用经过原模型、权重、生成设置和全部分片校验的真实贪心生成。','',
           '| Benchmark | 题数 | 主指标 | Native | LoopCD Fixed | LoopCD Adaptive | Mixed | T1 | T2 | T2−Native pp [95%] | T2−最佳 LoopCD pp [95%] | T2−T1 pp [95%] | T2−Mixed pp [95%] | T2 耗时 / 显存 |',
           '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: |']
    for task,label in NAMES.items():
        generation=task=='mmlu_pro'
        mode='generated' if generation else 'candidate'
        paths={arm:root/'predictions'/f'{task}__{arm}__k4__{mode}.jsonl' for arm,root in roots.items()}
        receipt=paths['t2'].with_suffix('.json')
        if not receipt.exists():
            continue
        t2_info=read(receipt)
        if generation:
            source,ids,ref,baseline_cost=archived_mmlu(scorer,a,lock,t2_info['runtime'],True)
            keys=('correct','strict_format','strict','parseable')
            primary='correct'
        else:
            panel=scorer.check_archive(a,scorer_impl,t2_info['runtime'],task,index)
            source=panel.public
            ids=[r['id'] for r in panel.requests]
            ref={arm:[candidate_row(r,name) for r in panel.references] for arm,name in METHODS.items()}
            baseline_cost=dict(joint_archived_scoring=panel.public['archived_joint_scoring_cost'])
            keys=('acc','acc_norm')
            primary=lock['primary'][task]
        paired_arms={arm:checked_arm(path,seals[arm],locks[arm],arm,ids,t2_info['plan']) for arm,path in paths.items()}
        own,cost,_=paired_arms['t2']
        for arm in ('mixed','t1'):
            ref[arm],baseline_cost[arm],_=paired_arms[arm]
        if (len(ids)!=len(set(ids)) or len(own)!=lock['expected'][task] or
                any(x['gold']!=y['gold'] for refs in ref.values() for x,y in zip(own,refs,strict=True))):
            raise ValueError(f'Unmatched population or answers for {task}')
        paired_analysis.METRICS.update(keys)
        pair={arm:{k:v for k,v in paired_analysis.compare(ref[arm],own).items() if k in keys} for arm in BASELINES}
        best=max(('fixed','adaptive'),key=lambda arm:pair[arm][primary]['reference'])
        seconds=cost['seconds'] if generation else cost['elapsed_seconds']
        peak=cost['peak_allocated_bytes']
        for arm,stats in pair.items():
            for metric,p in stats.items():
                records.append(dict(benchmark=task,n=p['n'],metric=metric,primary=metric==primary,reference=arm,
                                    t2=p['controller'],baseline=p['reference'],delta=p['delta'],rescued=p['rescued'],
                                    harmed=p['harmed'],ci_low=p['paired_ci95'][0],ci_high=p['paired_ci95'][1],
                                    mcnemar=p['exact_mcnemar'],seconds=seconds,peak_cuda_bytes=peak))
        results[task]=dict(n=len(ids),metric=primary,source=source,paired=pair,best_loopcd=best,
                           t2_cost=cost,baseline_cost=baseline_cost,
                           rows_sha256={arm:arm_result[2]['rows_sha256'] for arm,arm_result in paired_arms.items()})
        values=[pct(pair[arm][primary]['reference']) for arm in BASELINES]
        values.append(pct(pair['native'][primary]['controller']))
        differences=[contrast(pair[arm][primary]) for arm in ('native',best,'t1','mixed')]
        cells=' | '.join([*values,*differences,f'{seconds:.2f} s / {peak/2**30:.2f} GiB'])
        line=f'| {label} | {len(ids):,} | {primary} | {cells} |'
        lines.append(line)
        print(line,flush=True)
    total=sum(row['n'] for row in results.values())
    if a.final and (len(results)!=8 or total!=43769):
        raise ValueError(f'Incomplete T2 report: {len(results)} tasks, {total} samples')
    if not records:
        return
    scorer.json_write(a.out/'RESULTS.json',dict(freeze_sha256=seals['t2'],reference_freeze_sha256={arm:seals[arm] for arm in ('mixed','t1')},
                      completed_tasks=len(results),completed_questions=total,results=results))
    temp=a.out/'RESULTS.csv.partial'
    with temp.open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temp.replace(a.out/'RESULTS.csv')
    lines.extend(['',f'已核验 {total:,} 题。差值单位为百分点，区间为逐题配对 95% 区间。候选评分耗时为本轮批次推理总计，MMLU-Pro 的 T2 耗时和显存为经校验的历史记录。显存口径为 CUDA 峰值已分配显存。',
                  '', '[救回／退化统计和成本](RESULTS.csv) · [完整 JSON](RESULTS.json) · [冻结回执](FREEZE.json)',''])
    (a.out/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('out','mixed','t1','model','minimal','stage2','unified','followup'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    run(parser.parse_args())
