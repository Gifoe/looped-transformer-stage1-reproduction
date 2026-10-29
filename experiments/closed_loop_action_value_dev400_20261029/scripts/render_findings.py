"""Render a development research note from actual audited outputs."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
a=load(ROOT/'results/ANALYSIS.json');r=a['full400'];s=r['summary_vs_adaptive'];sub=a['exclude_implementation_pilot392'];budget=load(ROOT/'results/BUDGET.json')
names={'adaptive':'Adaptive LoopCD','fixed_-1.0':'固定完全回退','fixed_-0.5':'固定部分回退','fixed_0.5':'固定外推','confidence':'置信度控制','terminal_only':'探针选择、仅末端扰动','probe':'输入探针闭环控制','oracle':'Oracle 上界'}
lines=['# 400 题闭环动作价值开发实验','',
'本实验回答输入探针变量是否能选择有益的真实状态干预。所有题来自已暴露的 OpenBookQA discovery；这是开发证据，不是新的独立测试。前 8 题曾用于实现试验，另报排除它们后的 392 题结果。',
'','## 实际准确率与配对纠错','',
'| 策略 | 正确 / 400 | Accuracy | 纠错 | 损害 | 相对 Adaptive 增益 pp [95% CI] |',
'|---|---:|---:|---:|---:|---|']
for k in names:
    x=s[k];lo,hi=x['ci95_pp'];lines.append(f"| {names[k]} | {x['correct']} | {x['accuracy_percent']:.2f}% | {x['corrections']} | {x['degradations']} | {x['delta_pp']:+.3f} [{lo:.3f}, {hi:.3f}] |")
lines+=['','Oracle 是使用 gold 后得到的不可部署上界。表中主指标为官方继承的字符长度归一化完整候选似然准确率。',
'','## 中间变量是否包含动作收益信息','',
f"控制器动作次数：{r['selected_action_counts']}；有足够输入探针的题数为 {r['probe_available']}/400。推理时不使用 gold，同一个 prompt 决策用于所有候选。",
'','| 状态动作 | 有益 / 有害事件 | 探针收益 AUC [95% CI] | 与 gold 候选归一化 log-weight 收益的 Spearman 相关 |',
'|---|---|---|---|']
for key,label in [('-1.0','完全回退'),('-0.5','部分回退'),('0.5','外推')]:
    x=r['signals'][key];auc='不可估计（缺少两类事件）' if x['auc'] is None else f"{x['auc']:.3f} [{x['auc_ci95'][0]:.3f}, {x['auc_ci95'][1]:.3f}]"
    rho=x['spearman_normalized_gold_logweight'];lines.append(f"| {label} | {x['beneficial']} / {x['harmful']} | {auc} | {'不可估计' if rho is None else f'{rho:.3f}'} |")
lines+=['','AUC 只在该动作会改变正确性的非零事件上计算；置信区间为未作跨动作校正的分层 bootstrap，属于探索性诊断。不能将 AUC 或 log-weight 相关性当作最终准确率收益，也不能将两组相关输入 token 称为独立证据。',
'','## 选择信息和反馈作用','',
f"打乱动作分配、严格保留动作频率及探针可用性的 10,000 次随机对照，平均答对 {r['random']['mean_correct']:.3f} 题，95% 区间 {r['random']['correct_interval95']}；实际控制器答对 {r['random']['observed_correct']} 题，上尾置换 p={r['random']['upper_tail_p']:.6f}。",
'','控制器对六个预设比较对象的 Holm 校正 p：']
for k,p in r['holm_six_p'].items():lines.append(f"- {names[k]}：{p:.6f}")
lines+=['',f"预设开发控制器筛选：{'通过' if r['passes_development_controller_screen'] else '未通过'}。这个筛选即便通过，也只能支持登记新的独立确认实验。",
f"排除前 8 题后：基线 {sub['summary_vs_adaptive']['adaptive']['correct']}/392，控制器 {sub['summary_vs_adaptive']['probe']['correct']}/392，净增 {sub['summary_vs_adaptive']['probe']['net']} 题；筛选{'通过' if sub['passes_development_controller_screen'] else '未通过'}。不能在 400 和 392 两个结果中择优报告。",
'','## 成本与实现完整性','',
f"串行回放成本估计：控制器 / 基线为 {r['timing']['controller_to_baseline_ratio']:.3f} 倍。决策耗时包含额外置信度诊断，零动作分支已扣除末端扰动诊断；这不是优化后的部署性能或算力节省证明。",
f"本次所有工作进程、失败尝试及数值审计合计保守计费 {budget['study_charge_seconds']/3600:.4f} GPU 小时；累计新增研究用量 {budget['total_new_research_seconds']/3600:.4f}/8 小时。",
'',
'首次运行 52 条记录后，被 1e-5 绝对分数一致性检查拦住。单独审计定位到读出位置打包差异；零反馈状态及答案一致。修正位置打包后，边界两题与原分数完全相同，保留原阈值并完整重跑 400 题。失败记录、诊断脚本的一次参数传递错误和全部计费收据均保留。控制器、状态动作、模型及统计方案没有因结果而修改。',
'','全部动作只改变第 2 轮及之后的状态，未改变第 1 轮；主干保持冻结。原实验与原型文件哈希由独立本地审计核验。没有 early stop；撤回已经执行的计算不节省其成本。',
'','## 研究判断','']
if r['passes_development_controller_screen']:
    lines+=['开发结果支持继续检验这一闭环变量，但尚未证明跨任务有效或超过现有方法。下一步只能在配置固定后使用未暴露的独立任务确认，不能在旧 SciQ/PIQA/ARC/Qwen 结果上继续选规则。']
else:
    lines+=['当前训练免费输入探针控制器没有通过预设开发标准。实现了“变量进入控制环”不等于发现了可靠质量变量；其是否有动作收益信息应以以上事件数、区间和随机对照判断。该结果不否定所有闭环变量，但不支持把这一版本作为有效方法发表或直接进行独立确认。任何后续轻量监督都必须使用实际状态干预的收益标签，且不能靠本结果选择新阈值或翻转符号。']
diag_path=ROOT/'results/ROLLBACK_DIAGNOSTIC.json'
if diag_path.exists():
    d=load(diag_path)
    lines+=['','## 主分析后的定向诊断：排序与校准不同','',
    f"完全回退的输入探针增益在全部 {d['n_available']} 道探针充分的题上均不为正，最大值也只有 {d['proxy_max']:.6f} nat/token。因此当前正值门槛必然不会选择完全回退。有益回退事件的探针值中位数为 {d['beneficial_median']:.3f}，有害事件为 {d['harmful_median']:.3f}，存在需要新数据验证的排序线索。",
    f"同一 23 个非零事件上，预设正向 margin 变化的 AUC 为 {d['margin_change_auc']:.3f}，探针 AUC 为 {d['proxy_auc']:.3f}；未校正的配对 bootstrap AUC 差区间为 [{d['paired_auc_difference_ci95'][0]:.3f}, {d['paired_auc_difference_ci95'][1]:.3f}]。这是主分析后的探索性比较，不能证明条件独立的信息增益。训练后的置信度控制可从训练标签学习相反方向，必须公平地一并校准，不能用未校准的弱控制夸大探针优势。",
    '原始探针响应 S 的单位是已知输入 token 的损失变化，不能直接当成最终准确率动作价值 V。下一项有界检验应在不同训练题上用实际状态干预的收益校准 S→V，再检验未参与拟合的题；不在这 400 题上移阈值、换符号或宣称已得到可靠控制器。']
lines+=['','图与原始记录见 `figures/`、`results/ANALYSIS.json` 和 `results/development_packed.jsonl`。所有最终准确率主张尚未完成独立确认。','']
(ROOT/'FINDINGS_ZH.md').write_text('\n'.join(lines),encoding='utf8')
