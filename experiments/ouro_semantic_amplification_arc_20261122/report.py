"""Render result tables and answer the five research questions from audited files."""

import argparse
import json
from pathlib import Path


DEPTHS=(1,2,3,4)
ARMS={'magnitude':'Magnitude-only','direction':'Direction + Magnitude',
      'completion':'Semantic Completion','shuffled':'打乱教师'}


def load(root,name):
    return json.loads((root/f'{name}.json').read_text(encoding='utf-8'))


def interval(x):
    y=x['ci95_pp']
    return f"{x['delta_pp']:+.2f} [{y[0]:+.2f}, {y[1]:+.2f}]"


def score(row,n):
    return f"{row['correct']}/{n} ({row['pct']:.2f}%)"


def variants(metrics,method):
    return [metrics['methods'][f'{method}_k{k}'] for k in DEPTHS]


def frontier(metrics,arms):
    peak=-1
    output=[]
    for k in DEPTHS:
        variants=[a for a in metrics['methods'] if a.endswith(f'_k{k}') and arms(a)]
        top=max(metrics['methods'][a]['correct'] for a in variants)
        if top>peak:
            output.append((k,top,[a for a in variants if metrics['methods'][a]['correct']==top]))
            peak=top
    return output


def render(root):
    full=load(root,'METRICS')
    metrics=full['metrics']
    n=full['n']
    v=load(root,'VERIFICATION')
    freeze=load(root,'FREEZE')
    cost=load(root,'COST')
    historic_cost=load(root,'HISTORICAL_COST')
    failure=load(root,'FAILURES')
    geometry=load(root,'GEOMETRY_AUDIT')
    stamp=load(root,'TEST_INPUT')
    chosen=freeze['chosen']
    winner=max((a for a in ARMS if a!='shuffled'),key=lambda a:(sum(chosen[a]['metrics']['acc_norm'].values()),
                                                              sum(chosen[a]['metrics']['acc'].values())))
    mins,remainder=divmod(v['gpu_seconds'],60)
    base=metrics['acc_norm']
    raw=metrics['acc']
    win_rows=variants(base,winner)
    win_raw=variants(raw,winner)
    lines=['# 完整 ARC-Challenge 上的 Semantic Amplification','',
           f'冻结 Ouro-1.4B Base，使用一张 {stamp["runtime"]["gpu"]}。实际台账累计 {mins} 分 {remainder} 秒。官方 Test 全部 {n:,} 题、{stamp["candidates"]:,} 个完整候选，核验 {v["total_new_scores"]:,} 个新候选分数。',
           f'按 Validation 的四预算 `acc_norm` 总正确数，三种方法中选出的整体配置为 **{ARMS[winner]}**。其 Test 四个预算 `acc_norm` 净变化依次为 '+', '.join(f'{a["vs_native"]["delta_pp"]:+.2f} pp' for a in win_rows)+'。',
           '', '## 方法与证据来源','',
           '借位先前在第 15 层找到了对整答有作用的轮间交互项，且等幅匹配表明幅度可决定写入作用。先前的查表结果显示有效位置可随任务改变。ARC 采用历史原生终态和分支写入接口，在物理第 19–24 层分别汇总 Attention 与 MLP 两支路，两段各三层。当前位置取完整候选每个需预测 token 的前一位置。 [借位与写回](../ouro_causal_semantic_completion_20261122/REPORT_ZH.md) · [原生状态计算](../ouro_native_state_computation_20261121/REPORT_ZH.md)',
           r'令末六层实际写入和为 $w_k$，当前归一化前残差为 $h_k$。编码器把当前残差、两段 Attention、两段 MLP 和历史残差差值各投影到 8 维，输入共享门控。深度作为一个数值特征。写入为 $h_k^prime=h_k+d[g_phi(x_k)w_k+u_phi(x_k)]$，$gin[-0.5,0.5]$，$uperp w_k$ 且 $|u|leq0.5|w_k|$。$d$ 是跨预算共享的验证剂量。修改后只用一次原生 RMSNorm 与词表头。',
           'Magnitude-only 约束正交更新为零。Direction + Magnitude 从训练标签的完整候选排序目标学习正交分量。Semantic Completion 在相同架构和参数量上加入下一轮与本轮的分支写入差监督，先投影至当前写入的正交空间，并匹配尺度上限。此监督只用于 Train 的 K1–K3。打乱教师保持轮次和目标范数，在不同题之间置换监督方向。正式推理只运行指定深度，读取当前及已经运行的轮次。',
           '训练把 80% 权重给字符归一化候选交叉熵，20% 给未归一化的候选交叉熵。训练和评分沿用完整词表归一化，再求每个候选的 token 对数概率总和。`acc_norm` 除以原候选字符数。模型参数始终冻结。',
           '', '## Validation 决策与锁定','',
           '两轮训练使用不同的学习率及对应教师权重，每轮三轮训练遍历。候选训练题序和四个深度的分配规律一致。随后用第三轮在同一 299 题 Validation 比较共享剂量 0.50、0.75、1.00。优先四预算的 `acc_norm` 总正确数，次序为最差深度净增益、`acc` 总正确数。打乱教师使用真实教师所选的训练设置和剂量。',
           '', '| 方法 | 训练轮次 | 选中 Epoch | 共享剂量 | K1/K2/K3/K4 正确数（acc_norm /299） | 可训练参数 |',
           '|---|---:|---:|---:|---|---:|']
    for name,label in ARMS.items():
        c=chosen[name]
        w='/'.join(str(c['metrics']['acc_norm'][f'k{k}']) for k in DEPTHS)
        lines.append(f"| {label} | {c['round']} | {c['selected_epoch']} | {c['dose']:.2f} | {w} | {freeze['matched_parameters'][name]:,} |")
    lines.extend(['','官方 Train 为 1,119 题，Validation 为 299 题。历史实验已经在同一完整 Test 上运行过，逐题公开分数被用作只读参照。本轮冻结方法和评分规则后才由新方法评测 Test。历史参照按原题 ID、选项、字符长度、token 长度与每候选输入长度核对。','',
                  '## 完整 Test 结果',''])
    for metric,tab in metrics.items():
        lines.extend([f'### {metric}', '',
                      '| 深度 | 历史 Native | 历史 Test 最优 LoopCD | Magnitude-only | Direction + Magnitude | Semantic Completion | 打乱教师 |',
                      '|---|---:|---:|---:|---:|---:|---:|'])
        for k in DEPTHS:
            arms=tab['methods']
            native=score(arms[f'native_k{k}'],n)
            lc=tab['loopcd_historical_best'].get(f'k{k}')
            values=[score(arms[f'{a}_k{k}'],n) for a in ARMS]
            hist=f'{score(arms[lc],n)} `{lc}`' if lc else '—'
            lines.append('| '+' | '.join([f'K{k}',native,hist,*values])+' |')
        lines.extend(['', '| 深度与方法 | 比 Native 救回/退化 | 配对差 pp [95% CI] | McNemar p（四深度 Holm） | 比最优 LoopCD 救回/退化 | 配对差 pp [95% CI] |',
                      '|---|---:|---|---:|---:|---|'])
        for name,label in ARMS.items():
            for k,row in zip(DEPTHS,variants(tab,name),strict=True):
                a=row['vs_native']; b=row.get('vs_loopcd_historical_best')
                other=f"{b['rescued']}/{b['damaged']} | {interval(b)}" if b else '— | —'
                lines.append(f"| K{k} {label} | {a['rescued']}/{a['damaged']} | {interval(a)} | {a['holm_four_depth_p']:.3g} | {other} |")
        lines.append('')
    lines.extend(['Test 最优 LoopCD 列使用此前发布的 Test 结果中同预算各 LoopCD 变体的最高正确数。`METRICS.json` 同时给出仅由历史 Validation 锁定的 LoopCD 和其他历史变体的逐题配对统计。所有救回和退化以同一题的二元正误比较。','',
                  '## 消融与失败样本','',
                  '下表是 `acc_norm` 的配对差。组件消融使用同一选型、同一剂量。随机旋转还保留原生幅度、整体更新范数和修改后残差范数，仅改变正交分量在剩余子空间内的方向。',
                  '', '| 比较（后者为完整配置） | K1 | K2 | K3 | K4 |', '|---|---|---|---|---|'])
    for a,row in full['ablation']['acc_norm'].items():
        lines.append('| '+' | '.join([f'`{a}`',*(interval(row[f'k{k}']) for k in DEPTHS)])+' |')
    err=max(v for vector in geometry['max_residuals'].values() for v in vector)
    lines.extend(['',f'随机方向控制的三项几何约束最大绝对误差为 {err:.3g}。',
                  '', '| 深度 | 救回 / 退化 | 救回时的选项长度变化中位数 | 退化时的选项长度变化中位数 | 救回/退化的正交更新尺度（相对原生写入） |',
                  '|---|---:|---:|---:|---:|'])
    def datum(group,field):
        return f"{group['medians'][field]:+.3f}" if group['n'] else '—'
    for k in DEPTHS:
        f=failure[f'{winner}_k{k}']['cohorts']
        a,b=f['rescued'],f['damaged']
        lines.append(f"| K{k} | {a['n']}/{b['n']} | {datum(a,'delta_length_vs_native')} | {datum(b,'delta_length_vs_native')} | {datum(a,'direction_on_pred')} / {datum(b,'direction_on_pred')} |")
    for k in DEPTHS:
        f=failure[f'{winner}_k{k}']['examples']
        if f['damaged']:
            e=f['damaged'][0]; a=e['choices']
            lines.append(f"\nK{k} 首条退化案例 `{e['id']}`。原生选择 `{a[e['native']]}`，新方法选择 `{a[e['prediction']]}`，正确候选为 `{a[e['gold']]}`。")
    lines.extend(['','## 真实计算成本与前沿','',
                  '同一张 4090 D 上，逐法逐深度独立运行完整候选。时间包括模型前向、捕获、写回与候选求和，排除模型加载和独立审计。历史计时来自 RTX 4090，将硬件差异与同预算的准确率比较一并保留。',
                  '', '| 深度 | 方法 | 全量评分秒数 | 候选批中位 ms | 峰值 GiB | 实际填充 Token × Loop |',
                  '|---|---|---:|---:|---:|---:|'])
    for k in DEPTHS:
        h=historic_cost[f'native_k{k}']
        lines.append(f"| K{k} | 历史 Native（4090） | {h['wall_s']:.2f} | — | {h['peak_allocated_bytes']/2**30:.2f} | — |")
        for name,label in ARMS.items():
            x=cost[f'{name}_k{k}']
            lines.append(f"| K{k} | {label} | {x['seconds_full_split']:.2f} | {x['median_batch_ms']:.2f} | {x['peak_allocated_bytes']/2**30:.2f} | {x['padded_token_uses']:,} |")
    history=lambda a:a.startswith(('native_','loopcd_','laststep_','secant_'))
    new=lambda a:history(a) or any(a.startswith(s+'_k') for s in ('magnitude','direction','completion'))
    for metric,tab in metrics.items():
        for label,scope in (('历史','history'),('加入三种新方法','new')):
            seq=frontier(tab,history if scope=='history' else new)
            points=' · '.join(f'K{k} {v}/{n} `{",".join(a)}`' for k,v,a in seq)
            lines.append(f'\n{metric} 的{label}块预算前沿。{points}。')
    lines.extend(['','| Validation 锁定方法的深度 | 历史多一步最强配置（acc_norm） | 配对差 pp [95% CI] |',
                  '|---|---|---|'])
    for k,row in enumerate(win_rows[:3],1):
        c=row['vs_historical_next_depth_best']
        lines.append(f"| K{k} | `{c['reference']}` | {interval(c)} |")
    lines.extend(['','块预算前沿按执行的原生 Loop 数比较。当前写回还消耗特征提取和低秩增量开销。同卡 Native 与同卡 LoopCD 本轮未重跑，因此不能把跨硬件计时提升声明为严格的端到端 Pareto 支配。','',
                  '## Q1–Q5',''])
    mm=[a['vs_native']['delta_pp'] for a in win_rows]
    rr=[a['vs_native']['delta_pp'] for a in win_raw]
    lines.append(f'**Q1　是否提高四预算准确率？** Validation 锁定的 {ARMS[winner]} 在 `acc_norm` 的净增益为 '+
                 ' / '.join(f'{x:+.2f}' for x in mm)+' pp，在 `acc` 为 '+' / '.join(f'{x:+.2f}' for x in rr)+' pp。')
    def effects(key):
        stats=full['ablation']['acc_norm'][key]
        return ' / '.join(f"{stats[f'k{k}']['delta_pp']:+.2f}" for k in DEPTHS)
    lines.extend(['',f"**Q2　幅度还是方向？** 方向学习相对仅幅度变化为 {effects('direction_vs_magnitude')} pp。相对等几何的旋转方向为 {effects('direction_vs_rotated_same_geometry')} pp。",
                  '',f"**Q3　同一个方法能否跨深度？** 四个深度使用同一参数和剂量。`acc_norm` 正增益覆盖 {sum(a>0 for a in mm)}/4 个深度，`acc` 覆盖 {sum(a>0 for a in rr)}/4 个。",
                  '',f"**Q4　是否超过 LoopCD？** 与历史同深度最强 LoopCD 配对时，`acc_norm` 正增益覆盖 {sum(a['vs_loopcd_historical_best']['delta_pp']>0 for a in win_rows[1:])}/3，`acc` 覆盖 {sum(a['vs_loopcd_historical_best']['delta_pp']>0 for a in win_raw[1:])}/3。K1 比较历史原生。前沿和同卡限制见上表。",
                  '',f"**Q5　下一轮监督是否识别了可迁移计算？** 真实下一轮监督相对打乱教师的 `acc_norm` 净变化是 {effects('completion_vs_shuffled')} pp。只有在这一控制下仍有准确率增益时，才可把结果归因于同题跨轮目标。",
                  '', '逐题分数在 `TEST_PREDICTIONS.jsonl` 与 `GEOMETRY_PREDICTIONS.jsonl`，逐题配对选择在 `TEST_PAIRED.jsonl`。验证迭代轨迹与冻结签名同时发布。', ''])
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    a.output.write_text(render(a.root),encoding='utf-8')
