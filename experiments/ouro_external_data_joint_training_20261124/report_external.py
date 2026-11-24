"""Render final findings from verified results without substituting proxy metrics."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from task_data import load


def data(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pct(x):
    return f'{100*x:.2f}%'


def contrast(row):
    low,high=row['paired_ci95']
    return f"{100*row['delta']:+.2f} [{100*low:+.2f}, {100*high:+.2f}]"


def report():
    summary=data('results/RESULTS.json')
    select=data('results/DEVELOPMENT_SELECTION.json')
    verdict='达到本轮预先定义的主要验收条件' if summary['achieved_primary_observed_and_confirmed'] else '尚未达到全项兼容目标'
    table=['| 任务 | 主指标 | n | Native | 单一 Rank-1 | 对 Native 差值 pp [95%] | 救回／退化 | 最强历史 Rank-1 | 差值 pp [95%] |',
           '| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |']
    for row in summary['primary']:
        n=row['native']; s=row['paired_vs_best']
        table.append(f"| {row['dataset']} | {row['metric']} | {row['n']} | {pct(n['reference'])} | {pct(n['controller'])} | {contrast(n)} | {n['rescued']}／{n['harmed']} | {row['best_reference']} {pct(s['reference'])} | {contrast(s)} |")
    selected=summary['selected']
    recipe=selected['recipe']
    sources='CSQA 1,000、GSM8K 2,000 与 '+('OBQA 1,000' if recipe in 'AB' else 'ScienceQA 1,000')
    selection=f"A/B 初筛保留 {select['objective_winner']}，扩样入围 {', '.join(select['finalists'])}。最终选定 {selected['arm']}。"
    seed_rows=['| 配方 | 种子 | 实际 epoch | 明确退化端点 | 最弱 Native 差 pp | 最弱强参照差 pp |',
               '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in select['seed_results']:
        seed_rows.append(f"| {row['recipe']} | {row['seed']} | {row['epoch']} | {row['damaged']} | {100*row['minimum_native']:+.2f} | {100*row['minimum_strong']:+.2f} |")
    transfer=['| 任务与模式 | K2 | K3 | K5 | K6 |','| --- | --- | --- | --- | --- |']
    for ds,mode,metric in (('arc_challenge','candidate','acc_norm'),('csqa','content','correct'),('obqa','content','correct'),
                           ('squad','span','correct'),('gsm8k','gsm_process','strict')):
        values={row['depth']:contrast(row['paired']['native'][metric]) for row in summary['coverage']
                if row['scope']=='transfer' and row['dataset']==ds and row['mode']==mode}
        transfer.append(f"| {ds} {mode} | " + " | ".join(values[k] for k in (2,3,5,6)) + " |")
    p=Path('README.md')
    existing=p.read_text(encoding='utf-8').split('\n## 单卡运行\n',1)[1]
    report_lines=['# Ouro 单一 Rank-1 外部数据联合训练','',
                  f'**{verdict}。** 选定配方使用 {sources}。{selection}模型 SHA-256 为 `{summary["final_sha256"]}`。所有任务和深度使用同一权重。','',
                  '## K4 完整主面板','',*table,'',
                  '主指标下降和区间各自报告。区间包含零的差值仍有可能为负。两套候选协议分开汇总。MMLU-Pro 为实际自由生成。',
                  '', '## 同题深度迁移','',*transfer,'',
                  '各格为相对 Native 的百分点及配对区间。各深度使用按题号固定的同一 192 题。K4 全量结果见上表，逐题原始结果也支持将它限制在这一子集。',
                  '', '## 入围配方的全部种子','',*seed_rows,'',
                  '[完整结果和成本](results/RESULTS.csv) · [逐条件与逐种子冻结记录](results/RESULTS.json) · [数据划分](DATASET.md) · [有限干预及真实前缀](MECHANISM.md)','']
    p.write_text('\n'.join(report_lines)+'\n## 单卡运行\n'+existing,encoding='utf-8')


def mechanism():
    root=Path('results/mechanism')
    m=data(root/'SUMMARY.json')
    finite=['| 视图 | 组 | 剂量 | C 线性误差 | D 线性误差 | D−C 实际效用 [95%] |',
            '| --- | --- | ---: | ---: | ---: | --- |']
    for r in m['finite']:
        if r['depth']!=4:
            continue
        pair=r['paired_D_minus_C']
        lo,hi=pair['ci95']
        err=r['mean_absolute_linear_error']
        finite.append(f"| {r['view']} | {r['group']} | {r['dose']:.2f} | {err['C']:.4f} | {err['D']:.4f} | {pair['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    replay=load(root/'prefix_replay__k4.jsonl')
    by_origin=defaultdict(list)
    for row in replay:
        if row['position']==4:
            by_origin[row['origin']].append(row)
    text=['| 固定前缀来源，t=4 | n | D−C 提前结束对数几率 | D−C 注入答案标记后的首数字词元优势 |',
          '| --- | ---: | ---: | ---: |']
    if not by_origin:
        text.append('| 对齐前缀不足 | 0 | — | — |')
    for name,rows in by_origin.items():
        gains=[np.mean([row['D'][key]-row['C'][key] for row in rows]) for key in
               ('eos_log_odds','numeric_start_margin_after_counterfactual_marker')]
        text.append(f'| {name} | {len(rows)} | {gains[0]:+.4f} | {gains[1]:+.4f} |')
    notes=['# 同题解释监督与循环控制','',
           '比较固定种子、相同实际三轮训练和同一批 ScienceQA 原题的 C/D。共同独立题样本取自已锁定的开发池。数学真实生成与参考前缀用途分开，解释 token 较多的实际计算差保留在训练台账。','',
           *finite,'','上表的线性误差是逐题实际效用变化和局部预测之差的平均绝对值。配对区间以题为抽样单位。完整 K2–K6 数据见 JSONL 和 SUMMARY.json。','',
           *text,'','前缀按四种来源均能到达指定位置且尚未输出最终标记进行配对。注入标记仅用于诊断，真实生成时不访问金标、其他控制器或未来状态。','']
    Path('MECHANISM.md').write_text('\n'.join(notes),encoding='utf-8')


if __name__=='__main__':
    mechanism()
    report()
