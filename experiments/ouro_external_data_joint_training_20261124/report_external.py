"""Render final findings from verified results without substituting proxy metrics."""

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from select_external import ordering
from task_data import load


def data(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pct(x):
    return f'{100*x:.2f}%'


def contrast(row):
    low,high=row['paired_ci95']
    return f"{100*row['delta']:+.2f} [{100*low:+.2f}, {100*high:+.2f}]"


def audit_errors(summary):
    root = Path('results')
    refs = {row['dataset']: row['best_reference'] for row in summary['primary']}
    with (root/'ERRORS.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ('scope', 'dataset', 'depth', 'mode', 'id', 'metric', 'baseline', 'change', 'controller_failure', 'baseline_failure')
        output = csv.writer(handle)
        output.writerow(fields)
        for condition in summary['coverage']:
            scope, ds, depth, mode = (condition[key] for key in fields[:4])
            metric = condition['metric'] or {'candidate': 'acc_norm', 'gsm_process': 'strict'}.get(mode, 'correct')
            managed = scope == 'transfer' or scope == 'supplement' or ds not in {
                'arc_challenge', 'arc_easy', 'sciq', 'mmlu', 'hellaswag', 'winogrande', 'piqa', 'mmlu_pro'}
            folder = root / ('transfer/core' if scope == 'transfer' else 'test/core' if managed else 'test/benchmark')

            def rows(arm):
                return load(folder/f'{ds}__{arm}__k{depth}__{mode}.jsonl')

            ours = rows('joint')
            for arm in dict.fromkeys(['native', refs[ds]]):
                for a, b in zip(ours, rows(arm), strict=True):
                    if a['id'] != b['id']:
                        raise ValueError(f"Incorrect failure-audit pairing for {ds} K{depth} {mode}")
                    good, previous = a['metrics'][metric], b['metrics'][metric]
                    if good != previous:
                        output.writerow([scope, ds, depth, mode, a['id'], metric, arm, 'rescue' if good else 'damage',
                                         error_category(a, mode, metric), error_category(b, mode, metric)])


def error_category(row, mode, metric):
    measures = row['metrics']
    if measures[metric]:
        return ''
    if mode == 'candidate':
        return 'wrong_candidate'
    if mode == 'gsm_process':
        tags = [k for k in ('premature_eos', 'token_limit', 'repetitive') if measures.get(k)]
        tags.append('missing_final_marker_or_format' if measures.get('correct') else
                    'wrong_number' if measures.get('parsed') is not None else 'missing_number')
        return '|'.join(tags)
    if measures.get('parseable', measures.get('valid', True)) is False:
        return 'unparseable'
    if measures.get('label_conflict'):
        return 'label_content_conflict'
    return 'wrong_answer'


def report():
    summary=data('results/RESULTS.json')
    select=data('results/DEVELOPMENT_SELECTION.json')
    audit_errors(summary)
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
    explanation = '采用 ScienceQA 替换 OBQA。' if recipe in 'CD' else '保留 OBQA。'
    screened = {key: max(select['screen'][key], key=ordering) for key in 'ABCD'}
    source_table = ['| 配方 | 初筛轮次 | 明确退化端点 | 最弱 Native 差 pp | 最弱历史参照差 pp |',
                    '| --- | ---: | ---: | ---: | ---: |']
    for key, val in screened.items():
        source_table.append(f"| {key} | {val['epoch']} | {val['damaged']} | {100*val['minimum_native']:+.2f} | {100*val['minimum_strong']:+.2f} |")
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
                  f'**{verdict}。** 选定配方使用 {sources}。{selection}{explanation}模型 SHA-256 为 `{summary["final_sha256"]}`。所有任务和深度使用同一权重。','',
                  '## K4 完整主面板','',*table,'',
                  '主指标下降和区间各自报告。区间包含零的差值仍有可能为负。两套候选协议分开汇总。MMLU-Pro 为实际自由生成。',
                  '', '## 同题深度迁移','',*transfer,'',
                  '各格为相对 Native 的百分点及配对区间。各深度使用按题号固定的同一 192 题。K4 全量结果见上表，逐题原始结果也支持将它限制在这一子集。',
                  '', '## 四配方初筛','',*source_table,'',
                  'A/B 使用同题同序的 OBQA 混合数据，B 额外计算候选排序。C/D 在相同 ScienceQA 题目上比较短答与解释。初筛表使用同一开发子集，扩样后的全种子结果决定最终取舍。见 [C/D 逐题配对、曝光和训练成本](results/C_D_DEV_AUDIT.json)。',
                  '', '## 入围配方的全部种子','',*seed_rows,'',
                  '[完整结果和成本](results/RESULTS.csv) · [救回和退化题](results/ERRORS.csv) · [逐条件与逐种子冻结记录](results/RESULTS.json) · [数据划分](DATASET.md) · [有限干预及真实前缀](MECHANISM.md)','']
    p.write_text('\n'.join(report_lines)+'\n## 单卡运行\n'+existing,encoding='utf-8')


def mechanism():
    root=Path('results/mechanism')
    m=data(root/'SUMMARY.json')
    finite=['| 视图 | 组 | 剂量 | C／D 有符号局部导数 | C／D 有限效用变化 | C／D 符号一致率 | D−C 实际效用 [95%] |',
            '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for r in m['finite']:
        if r['depth']!=4:
            continue
        pair=r['paired_D_minus_C']
        lo,hi=pair['ci95']
        slopes = '／'.join(f"{r['local_signed_slopes'][a]['mean']:+.4f}" for a in 'CD')
        effects = '／'.join(f"{r['native_relative'][a]['mean']:+.4f}" for a in 'CD')
        agrees = '／'.join(pct(r['sign_agreement'][a]) for a in 'CD')
        finite.append(f"| {r['view']} | {r['group']} | {r['dose']:.2f} | {slopes} | {effects} | {agrees} | {pair['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
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
    trajectories=['| 深度 | 模型 | 严格数值正确 | 完整过程与正确答案 | 提前 EOS | 到达长度上限 |',
                  '| ---: | --- | ---: | ---: | ---: | ---: |']
    for depth, detail in m['self_prefix'].items():
        for arm, measures in detail['arms'].items():
            rates = [pct(measures[k]['value']) for k in ('strict', 'process_correct', 'premature_eos', 'token_limit')]
            trajectories.append(f"| {depth} | {arm} | " + ' | '.join(rates) + ' |')
    notes=['# 同题解释监督与循环控制','',
           '比较固定种子、相同实际三轮训练和同一批 ScienceQA 原题的 C/D。共同独立题样本取自已锁定的开发池。数学真实生成与参考前缀用途分开，解释 token 较多的实际计算差保留在训练台账。','',
           *finite,'','符号一致率按题比较实际效用变化和局部预测。配对区间以题为抽样单位。JSONL 和 SUMMARY.json 保存全部 K2–K6 度量和线性误差。','',
           *text,'','前缀按四种来源均能到达指定位置且尚未输出最终标记进行配对。注入标记仅用于诊断，真实生成时不访问金标、其他控制器或未来状态。','',
           *trajectories,'','过程结构由生成文本的格式规则判定。','']
    Path('MECHANISM.md').write_text('\n'.join(notes),encoding='utf-8')


if __name__=='__main__':
    mechanism()
    report()
