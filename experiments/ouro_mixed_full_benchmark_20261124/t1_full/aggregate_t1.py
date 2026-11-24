"""Calculate T1's paired differences on the complete frozen Mixed benchmark panel."""

import argparse
import csv
from pathlib import Path
import sys

from evaluate_benchmarks import candidate_row
from evaluate_core import verified
import paired_analysis


sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mixed_full'))
from aggregate_mixed import NAMES, METHODS, archived_mmlu, contrast, pct, read


def checked_arm(path, registry, freeze, arm, ids, planned):
    data=read(path.with_suffix('.json'))
    plan=data['plan']
    required=('depth','input_sha256','config','batch_size','capacity_tokens','max_new')
    if (plan['freeze_sha256'] != registry or plan['checkpoint_sha256'] != freeze['arms'][arm]['sha256']
            or any(plan.get(name) != planned.get(name) for name in required)):
        raise ValueError(f'Unmatched locked protocol: {path}')
    result,cost=verified(path,plan,ids)
    return result,cost,data


def run(a):
    sys.path[:0]=[str(a.unified),str(a.stage2/'code'),str(a.followup)]
    import seven_unified as scorer
    import stage2_score as scorer_impl

    scorer_impl.APPLE_BENCHMARKS=dict.fromkeys(scorer_impl.APPLE_BENCHMARKS,0)
    scorer.TASKS={name:(0,n,m) for name,(_,n,m) in scorer.TASKS.items()}
    own_lock=read(a.out/'FREEZE.json')
    mix_lock=read(a.mixed/'FREEZE.json')
    own_sha=scorer.sha(a.out/'FREEZE.json')
    mix_sha=scorer.sha(a.mixed/'FREEZE.json')
    if own_lock['mixed_protocol_sha256'] != mix_sha:
        raise ValueError('Different Mixed benchmark panel')
    for name in ('depth','expected','model_sha256','stage2_sha256','seven_requests_sha256','mmlu_pro_sha256','mmlu_pro_code_sha256'):
        if own_lock[name] != mix_lock[name]:
            raise ValueError(f'T1 and Mixed differ on {name}')
    index=read(a.stage2/'COMPARISON.json')
    results, records={}, []
    lines=['# 历史 Rank-1 T1 完整 K4 评测','',
           '同一套 4,096 参数 T1 用于全部任务。七项候选评分沿用 Mixed 的零样本请求；MMLU-Pro 复用经校验的 T1 原始全词表贪心生成。',
           '', '| Benchmark | 题数 | 主指标 | Native | LoopCD Fixed | LoopCD Adaptive | Mixed | T1 | T1−Native pp [95%] | T1−最佳 LoopCD pp [95%] | T1−Mixed pp [95%] | T1 耗时 / 显存 |',
           '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |']
    for task,label in NAMES.items():
        generation=task=='mmlu_pro'
        mode='generated' if generation else 'candidate'
        path=a.out/'predictions'/f'{task}__t1__k4__{mode}.jsonl'
        if not path.with_suffix('.json').exists():
            continue
        doc=read(path.with_suffix('.json'))
        if generation:
            source,ids,ref,baseline_cost=archived_mmlu(scorer,a,own_lock,doc['runtime'],True)
            keys=('correct','strict_format','strict','parseable')
            primary='correct'
        else:
            panel=scorer.check_archive(a,scorer_impl,doc['runtime'],task,index)
            source=panel.public
            ids=[r['id'] for r in panel.requests]
            ref={arm:[candidate_row(r,name) for r in panel.references] for arm,name in METHODS.items()}
            baseline_cost=dict(joint_archived_scoring=panel.public['archived_joint_scoring_cost'])
            keys=('acc','acc_norm')
            primary=own_lock['primary'][task]
        own,cost,_=checked_arm(path,own_sha,own_lock,'t1',ids,doc['plan'])
        other=a.mixed/'predictions'/f'{task}__mixed__k4__{mode}.jsonl'
        ref['mixed'],mixed_cost,mixed_doc=checked_arm(other,mix_sha,mix_lock,'mixed',ids,doc['plan'])
        if (len(ids)!=len(set(ids)) or len(own)!=own_lock['expected'][task] or
                any(x['gold']!=y['gold'] for refs in ref.values() for x,y in zip(own,refs,strict=True))):
            raise ValueError(f'Unmatched population or answers for {task}')
        paired_analysis.METRICS.update(keys)
        pair={arm:{k:v for k,v in paired_analysis.compare(rows,own).items() if k in keys} for arm,rows in ref.items()}
        seconds=cost['seconds'] if generation else cost['elapsed_seconds']
        peak=cost['peak_allocated_bytes']
        best=max(('fixed','adaptive'),key=lambda arm:pair[arm][primary]['reference'])
        for arm,stats in pair.items():
            for metric,p in stats.items():
                records.append(dict(benchmark=task,n=p['n'],metric=metric,primary=metric==primary,reference=arm,
                                    t1=p['controller'],baseline=p['reference'],delta=p['delta'],rescued=p['rescued'],
                                    harmed=p['harmed'],ci_low=p['paired_ci95'][0],ci_high=p['paired_ci95'][1],
                                    mcnemar=p['exact_mcnemar'],seconds=seconds,peak_cuda_bytes=peak))
        baseline_cost['mixed']=mixed_cost
        results[task]=dict(n=len(ids),metric=primary,source=source,paired=pair,best_loopcd=best,
                           t1_cost=cost,baseline_cost=baseline_cost,t1_rows_sha256=doc['rows_sha256'],
                           mixed_rows_sha256=mixed_doc['rows_sha256'])
        scores=[pct(pair[arm][primary]['reference']) for arm in ('native','fixed','adaptive','mixed')]
        scores.append(pct(pair['native'][primary]['controller']))
        differences=[contrast(pair[arm][primary]) for arm in ('native',best,'mixed')]
        cell=' | '.join([*scores,*differences,f'{seconds:.2f} s / {peak/2**30:.2f} GiB'])
        line=f'| {label} | {len(ids):,} | {primary} | {cell} |'
        lines.append(line)
        print(line,flush=True)
    total=sum(row['n'] for row in results.values())
    if a.final and (len(results)!=8 or total!=43769):
        raise ValueError(f'Incomplete T1 report: {len(results)} tasks, {total} samples')
    if not records:
        return
    scorer.json_write(a.out/'RESULTS.json',dict(freeze_sha256=own_sha,mixed_freeze_sha256=mix_sha,
                      completed_tasks=len(results),completed_questions=total,results=results))
    temp=a.out/'RESULTS.csv.partial'
    with temp.open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temp.replace(a.out/'RESULTS.csv')
    lines.extend(['',f'已核验 {total:,} 题。差值单位为百分点，95% 区间按题配对。候选项耗时为本轮推理批次总计，MMLU-Pro 的 T1 耗时和显存来自已校验的历史记录。显存为 CUDA 峰值已分配显存。',
                  '', '[救回／退化统计和成本](RESULTS.csv) · [完整 JSON](RESULTS.json) · [冻结回执](FREEZE.json)', ''])
    (a.out/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('out','mixed','model','minimal','stage2','unified','followup'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    run(parser.parse_args())
