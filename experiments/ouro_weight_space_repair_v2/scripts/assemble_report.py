"""Render the final Chinese report from complete, audited numerical artifacts."""
import csv
import json
from pathlib import Path
from runtime import ROOT,save


def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def csvrows(p):
    with (ROOT/p).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def signed(x):return f'{float(x):+.3f}'


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',default='results/REPORT_ASSEMBLY.json')
    options=parser.parse_args()
    assert (ROOT/options.receipt).resolve().is_relative_to(ROOT.resolve())
    a=read('results/FINAL_ANALYSIS.json');assert a['status']=='complete'
    audit=read('ARTIFACT_AUDIT.json');assert audit['status']=='passed'
    gate=read('results/SELECTED_CHECKPOINT_AUDIT.json');assert gate['status']=='passed'
    budget=read('logs/gpu_budget.json');assert budget['used_seconds']<=21600
    assert all(r['status'] in ['complete','failed'] for r in budget['runs'])
    selected=a['selected_edit'];cid=selected['selected_candidate_id']
    contrast={r['method']:r for r in a['contrasts']};plain=contrast['native_k4'];fixed=contrast['loopcd_k4_fixed']
    adaptive=contrast['loopcd_k4_adaptive'];search=csvrows('edit_search_results.csv')
    design=read('configs/CANDIDATE_DESIGN.json');data=read('configs/DATA_LOCK.json')
    frozen=read('configs/CANDIDATES_FROZEN.json');specs={s['id']:s for s in frozen['candidates']}
    groups=read('edited_weights/statistics/groups.json')['groups']
    wide=csvrows('final_comparison.csv')
    def verdict(c):
        delta=c['delta_accuracy_pp']
        if delta<0:
            if c['paired_bootstrap_95_high_pp']<0 and c.get('holm_p_two_primary_comparisons',1)<.05:
                return '准确率显著下降'
            return '点估计下降，差异不显著；未建立提升'
        if delta==0:return '准确率持平'
        if c['paired_bootstrap_95_low_pp']>0 and c.get('holm_p_two_primary_comparisons',1)<.05:
            return '本锁定回归基准支持提升'
        return '仅点估计上升，尚未建立可靠提升'
    text=f'''# Ouro V2：实际稀疏列与低秩权重编辑

已完成 14 个真实 BF16 权重候选、独立 selection、官方 validation 选定一个编辑版本，以及全部 **1,172 道 ARC-Challenge 测试题**。最终选择 **{cid}**。Edited Ouro K4 为 **{plain['edited_correct']}/1172（{plain['edited_accuracy_percent']:.3f}%）**，原始模型为 {plain['original_correct']}/1172（{plain['original_accuracy_percent']:.3f}%），差值 **{signed(plain['delta_accuracy_pp'])} pp**，结论：{verdict(plain)}。主组合 Edited + Fixed LoopCD 为 **{fixed['edited_correct']}/1172（{fixed['edited_accuracy_percent']:.3f}%）**，相对原始 Fixed LoopCD **{signed(fixed['delta_accuracy_pp'])} pp**，结论：{verdict(fixed)}。

这里的官方 test 曾在前序研究中使用，是方法冻结后的回归基准；本报告不将它称为新盲测。V1 原始 checkpoint、代码、数据和结果均保留，V2 没有沿用 V1 的“不编辑”停止条件。

## 1. 修改了哪些权重，为什么

从官方 train 的 384 道 discovery 题采集 K3、K4、K5、K6 的 MLP/Attention 输出投影前后统计。每个候选答案仅记录其全文续写的预测位置；先对 token 求均值，再对答案选项等权平均。保留全部题的逐选项循环状态和逐题分数，不只检查 V1 的少数 Correct→Wrong 案例。

独立按 K3→4、K4→5、K5→6 的正确答案 margin 变化排序，每个转移取最低 96 题和最高 96 题，使用固定 SHA256 规则破同分。margin 使用候选全文 log-likelihood 除以字符数后的正确选项分数减最大错误选项分数。各组类别如下：

| 转移 | C→W | W→C | C→C | W→W | 下降组 / 上升组 |
|---|---:|---:|---:|---:|---:|
'''
    for g in groups:
        c=g['transition_category_counts'];k1,k2=g['transition']
        text+=f"| K{k1}→K{k2} | {c['C_to_W']} | {c['W_to_C']} | {c['C_to_C']} | {c['W_to_W']} | 96 / 96 |\n"
    text+='''
每层构造下降组与上升组的 raw output 增量均值之差，转移权重预先固定为 0.5、0.25、0.25。以该差向量范数除以全体增量的 RMS 为 site score，选择两个 MLP 层和一个 Attention 层。rank1 是归一化差方向；rank4 在其正交补中加入下降组中心化增量的前三个主成分，再正交化。方向符号固定，退化秩不填充随机向量。这些是使用训练标签构造的统计方向，未使用梯度或优化器；不把它们直接称为已证实的“有害机制”。

MLP 列按“下降/上升组输入二阶矩增量差的正部 × 方向与该权重列的投影平方”排名；若该排名全零才回退到输入能量乘权重列范数，并记录回退情况。实际 site/column 排名、样本 ID、统计文件哈希和方向因子位于 `edited_weights/statistics/`。

本次选出的物理层（从 0 开始）为：'''
    text+=', '.join(f"{name}={layer}" for name,layer in frozen['selected_sites'].items())+'。\n\n'
    text+='''稀疏方向编辑为 `W[:,S] -= alpha * u @ (u.T @ W[:,S])`；稠密方向编辑为 `W -= alpha * U @ (U.T @ W)`。另有整列乘 0.5 的稀疏控制候选。Ouro 在输出投影后使用分支 RMSNorm，因此不能把整矩阵统一缩放等同于归一化后的残差阻尼；本轮直接改变列或方向，并在原始 BF16 参数中写入结果。所有循环调用共享同一个修改后的物理参数。未编辑 embedding、LM head、tokenizer 或 norm 权重，未引入 adapter、梯度、backward 或新网络。

这是对 [MIT Weight-Space Programming 公开实现](https://github.com/RikiyaT/lm_understand-logs) 的任务与架构适配，而非其数学基准或 Qwen 配置的逐项复现。原实现使用少量列上的方向分量修改；本轮没有照搬其层号与激进强度，而是用 Ouro 训练统计选层并固定小规模搜索。[公开方法说明](https://rikiyat.github.io/blog/weight-space-programming/)。

## 2. BF16 实际改变参数与精确重放

'''
    text+=f"选定 {cid} 实际改变 **{selected['actual_changed_parameters']:,} / {selected['total_unique_parameters']:,} = {100*selected['actual_changed_fraction_of_model']:.6f}%** 个 BF16 坐标。修改的实际 Frobenius 范数为 **{selected['actual_delta_frobenius']:.8g}**。计数比较的是量化后实际 tensor 位值，不把 rank1/rank4 的方向维数当成改变参数量。量化后的实际 delta 不保证仍然严格低秩。\n\n"
    text+='| 物理参数 | shape | 实际改变坐标 | ΔW Frobenius | 相对范数 | max abs ΔW |\n|---|---|---:|---:|---:|---:|\n'
    for e in selected['edits']:
        text+=f"| `{e['tensor_name']}` | {e['shape']} | {e['actual_changed_coordinates']:,} | {e['actual_delta_frobenius']:.7g} | {e['relative_delta_frobenius']:.7g} | {e['max_abs_delta']:.7g} |\n"
    text+='''
每个候选仅物化一次 BF16 目标张量，后续 screen、selection、validation、test 使用其精确替换 patch；不重复计算低秩 GEMM 后再量化。`selected_edit_manifest.json` 记录原始/编辑 tensor SHA256、文件 SHA256、实际坐标数及选择依据；`edited_weights/selected/` 保存精确替换 patch、完整 BF16 目标张量和方向因子。公开归档可还原这些文件。

真实 A01/B01 检查覆盖非零 BF16 修改、分数变化、精确恢复、K6 中间深度与独立 K3/K4/K5 一致，以及由编辑模型自身 K1/K4 状态独立重算两种 LoopCD，所有要求的分数误差为 0。最终选定 checkpoint 又在训练样本上通过所有参数 tensor 的保护检查、每个编辑参数跨六次循环调用对象/存储一致检查、先前训练分数重放、独立深度及 patch 再载入检查。检查范围和逐项结果见 `results/EDITED_PATH_GATE.json`、`results/SELECTED_CHECKPOINT_AUDIT.json`。

## 3. 数据隔离与验证集选择

'''
    text+=f"固定同一组 25-shot 示范，seed=2021，文件 SHA256：`{data['demonstrations_sha256']}`。384 道 discovery = V1 的 192 题 + 192 道之前未用的训练题；selection 保持独立 192 题；二者及示范按规范化问题文本排除重叠。\n\n"
    text+='''所有 14 候选首先在固定 64 道 discovery 子集筛查；按预先声明的家族配额，各家族前两名进入 selection192。selection 中各家族第一名加剩余最佳一名进入官方 validation299。最终验证集三选一，不再修改方向、列、强度或组合。排序严格为 K4 正确题数、K4 平均正确答案 margin、较少实际 BF16 改变数、候选 ID。K3/K5/K6 只记录，不改变选择规则。原模型没有最终选择资格，即使所有编辑都更差也选一个真实编辑模型。

只有写入 `VALIDATION_LOCK.json` 后才复制/读取 validation manifest；只有写入 `FINAL_LOCK.json` 后才复制/读取 test manifest。完整输入 SHA、代码 SHA、方向与实际 BF16 bundle SHA 贯穿每阶段逐题记录和完成收据。测试评估后才读取 V1 的逐题 test baseline，未按 test 反馈选择另一个候选或循环深度。

验证集候选：

| 候选 | K4 正确 / 299 | K4 acc_norm | 实际改变参数 | 是否选定 |
|---|---:|---:|---:|---|
'''
    for r in selected['validation_ranking']:
        v=r['metrics']['native_k4'];text+=f"| {r['candidate_id']} | {v['correct']} / 299 | {100*v['accuracy']:.3f}% | {r['actual_changed_parameters']:,} | {'是' if r['candidate_id']==cid else '否'} |\n"
    text+='''
## 4. 全量 1,172 题：Edited Ouro 是否改善

下表每格为正确题数（acc_norm）。LoopCD 只报告支持的 K4；NA 不代表估计值。

| 方法 | K3 | K4 | K5 | K6 |
|---|---:|---:|---:|---:|
'''
    for r in wide:
        cells=[]
        for k in [3,4,5,6]:
            c=r[f'K{k}_correct'];p=r[f'K{k}_accuracy_percent']
            cells.append('NA' if c=='NA' else f'{c} ({float(p):.3f}%)')
        label=r['method'].replace('fixed_loopcd','Fixed LoopCD').replace('adaptive_loopcd','Adaptive LoopCD')
        text+='| '+label+' | '+' | '.join(cells)+' |\n'
    text+='''
原始对照复用 V1 已在完全相同固定示范、逐题 prompt 和原生无 cache 调用下生成的 BF16 结果，本轮没有重复运行原始 test baseline。模型 revision、源权重哈希、环境与评分代码路径受记录约束，采集训练统计时对原始分数的两题八方法检查完全一致，最终逐题比较再次核验 ID、gold、prompt SHA、候选字符长度和 token 数。普通推理和 LoopCD 均由一个 K6 轨迹读取对应状态，不能据此声称分别测量了每个 K 的运行延迟。

评分使用 lm-eval 0.4.9 `acc_norm` 协议，完整候选答案的 teacher-forced log-likelihood 除以答案字符数，不是答案字母概率，也不是 token 长度归一化。BF16 的调用细节保留为 `model.model(input_ids=x, use_cache=False)`；不额外传全一 attention mask，以避免在固定 Transformers/SDPA 环境中改变数值路径。

## 5. Edited Ouro + LoopCD 是否胜过原始 LoopCD

主组合是同一份选定权重的 Fixed LoopCD，omega=0.5、K4、参考 K1；Adaptive cap1 是次要比较。二者只使用该编辑模型自身的 normalized K1/K4 状态，经 BF16 LM head 后以 FP32 组合 logits；未混入原始模型缓存，未按测试题调整系数。

| 同方法 Edited − Original | 纠错 | 退化 | 净题数 | Δ accuracy pp | 配对 bootstrap 95% CI pp | exact McNemar p | Holm 两主比较 p |
|---|---:|---:|---:|---:|---|---:|---:|
'''
    for m in ['native_k4','loopcd_k4_fixed','loopcd_k4_adaptive']:
        c=contrast[m];holm=c.get('holm_p_two_primary_comparisons');h='NA' if holm is None else f'{holm:.5g}'
        text+=f"| {m} | {c['corrections']} | {c['regressions']} | {c['net_correct']:+d} | {signed(c['delta_accuracy_pp'])} | [{c['paired_bootstrap_95_low_pp']:.3f}, {c['paired_bootstrap_95_high_pp']:.3f}] | {c['exact_mcnemar_p']:.5g} | {h} |\n"
    text+=f"\n普通 K4：{verdict(plain)}；Fixed 主组合：{verdict(fixed)}；Adaptive 次要比较差值 {signed(adaptive['delta_accuracy_pp'])} pp。配对 bootstrap 为 50,000 次题级重采样的 95% percentile 区间，采用与逐题重采样严格等价的纠错/退化/不变类别 multinomial 实现。McNemar 是 discordant pairs 上双侧 exact binomial 检验。两项预定主比较进行 Holm 校正；完整六项比较的校正值另列 `paired_statistics.csv`。区间是逐项区间，不是同时置信带。\n"
    text+='''
## 6. 跨循环深度与选择外样本

| 普通推理深度 | 纠错 / 退化 | 净题数 | Δ accuracy pp | 配对 95% CI pp |
|---|---:|---:|---:|---|
'''
    for m in ['native_k3','native_k4','native_k5','native_k6']:
        c=contrast[m];text+=f"| {m} | {c['corrections']} / {c['regressions']} | {c['net_correct']:+d} | {signed(c['delta_accuracy_pp'])} | [{c['paired_bootstrap_95_low_pp']:.3f}, {c['paired_bootstrap_95_high_pp']:.3f}] |\n"
    values=[contrast[f'native_k{k}']['delta_accuracy_pp'] for k in [3,4,5,6]]
    consistency='四个深度的点估计均为正；这不自动等于四个深度均有显著改善。' if all(v>0 for v in values) else '四个深度的点估计均下降。' if all(v<0 for v in values) else '各深度没有一致的正向点估计，不能声称跨循环深度稳定提升。'
    text+='\n'+consistency+'\n\n'
    text+='| 数据用途 | n | 普通 K4 Edited / Original | Δ pp | Fixed 组合相对原 Fixed Δ pp |\n|---|---:|---:|---:|---:|\n'
    for split,label in [('screen','训练 discovery 筛查'),('selection','训练 selection（用于晋级）'),('validation','官方 validation（用于选定）'),('test','历史使用的锁定 test')]:
        c=next(x for x in a['generalization'] if x['scope']==split and x['method']=='native_k4')
        f=next(x for x in a['generalization'] if x['scope']==split and x['method']=='loopcd_k4_fixed')
        text+=f"| {label} | {c['n']} | {c['edited_correct']} / {c['original_correct']} | {signed(c['delta_accuracy_pp'])} | {signed(f['delta_accuracy_pp'])} |\n"
    overlaps={k:v for k,v in audit['normalized_question_overlap'].items() if v}
    if overlaps:
        text+='\n规范化问题文本的跨集合重叠：'+', '.join(f'{k}={v}' for k,v in overlaps.items())+'。保留官方全量验证/测试集；重叠题不能视为独立于其来源训练题的新证据。详见 `results/SPLIT_OVERLAP_AUDIT.json`。\n'
    else:
        text+='\n本轮大小写/空白规范化问题文本检查未发现这些集合间重叠；这项检查不等于语义近重复检测。\n'
    text+='''
discovery 表只表示实际筛查的 64 题，不冒充 384 题编辑后全评；全部 384 题均用于前向方向发现。selection 与 validation 参与了模型选择，其增益不能当成独立泛化证据。test 的冻结后比较可评估此次修改在选择题之外的回归表现，但该集合曾被使用，且只有一个模型和一个任务，不能据此主张跨任务泛化或新失效机制。

![准确率与配对差值](figures/accuracy_by_loop.png)

## 7. 实际测试了哪些候选，失败幅度是多少

表中 screen 为 64 题、selection 为 192 题、validation 为 299 题。空缺表示未晋级，未补跑或推测结果。完整 K3–K6、两种 LoopCD、实际范数及逐阶段纠错/退化均在 `edit_search_results.csv`。

| 候选 | 家族 | 编辑摘要 | 实际改变 BF16 数 | screen K4 正确 (Δpp) | selection K4 正确 | validation K4 正确 |
|---|---|---|---:|---|---:|---:|
'''
    recipes={r['id']:r for r in design['candidates']}
    for candidate in specs:
        sr=next(r for r in search if r['candidate_id']==candidate and r['stage']=='screen')
        se=next((r for r in search if r['candidate_id']==candidate and r['stage']=='selection'),None)
        va=next((r for r in search if r['candidate_id']==candidate and r['stage']=='validation'),None)
        recipe=recipes[candidate]
        detail=recipe['mode']+', '+('Top'+str(recipe.get('columns',recipe.get('columns_each')))+' each' if 'columns' in recipe or 'columns_each' in recipe else 'rank'+str(recipe['rank']))
        detail+=', alpha='+str(recipe.get('alpha',recipe.get('alpha_each')))
        text+=f"| {candidate} | {specs[candidate]['family']} | {detail} | {int(sr['actual_changed_parameters']):,} | {sr['native_k4_correct']} ({signed(sr['delta_accuracy_pp'])}) | {se['native_k4_correct'] if se else 'NA'} | {va['native_k4_correct'] if va else 'NA'} |\n"
    text+='''
本实验的结论限于这 14 个预先固定的候选及选择过程。无论最终方向是正是负，都不能把本候选集结果升级为“所有稀疏/低秩免训练权重编辑有效或无效”。本轮不设计 Loop，不做新训练，也没有用 observational correlation 替代真实编辑评估。

## 8. 改答案例与交付

`paired_transitions.csv` 包含 1,172×6=7,032 条逐题同方法比较；`changed_answer_cases.csv` 包含所有预测改变，包括 Wrong→不同 Wrong。完整候选分数、字符归一化 margin、ID 和 prompt SHA 可回溯至逐题日志。预测与 gold 均为从 0 开始的选项索引，按源数据选项顺序排列。以下案例只是冻结后结果的展示，不参与参数或候选选择。

'''
    cases=csvrows('changed_answer_cases.csv')
    for method in ['native_k4','loopcd_k4_fixed']:
        for typ,label in [('C_to_W','退化'),('W_to_C','纠错')]:
            rr=[r for r in cases if r['method']==method and r['transition']==typ]
            rr=sorted(rr,key=lambda r:float(r['delta_margin']),reverse=typ=='W_to_C')[:2]
            for r in rr:
                text+=f"- {method} {label} `{r['id']}`：原预测 {r['original_prediction']} → 编辑预测 {r['edited_prediction']}，gold {r['gold']}，margin {float(r['original_margin']):.5f} → {float(r['edited_margin']):.5f}。问题：{r['question']}\n"
    text+=f'''
实际额外 GPU-worker 计费 **{budget['used_seconds']/3600:.4f} 小时**（{budget['used_seconds']:.3f} 秒），上限 6 小时；包括前向统计、加载、候选筛查、selection、validation、完整选定模型测试和最终正确性复验。CPU 归档/作图不占 GPU-worker 配额。没有耗费预算重跑 V1 的广泛标量损伤实验。

主要交付：

- `REPORT_ZH.md`、`accuracy_by_loop.pdf` 及可编辑 SVG/PNG。
- `edit_search_results.csv`、`candidate_edit_audit.csv`、`final_comparison.csv`、`paired_statistics.csv`、`paired_transitions.csv`、`changed_answer_cases.csv`、`generalization_by_split.csv`。
- `selected_edit_manifest.json`、冻结设计与阶段锁、实际 BF16 patch/目标张量与方向统计。
- 完整实验与独立重放脚本、逐题评分、完成收据、环境与预算日志、Astra 代码/统计审查。

远端完整原始逐题前向统计及全部候选权重保存在本实验目录。公开归档布局、哪些大文件保留远端以及校验/重放步骤见 `README.md` 和 `ARTIFACT_INDEX.json`；任何未上传的大体积诊断数据都会明确列出，不冒充已发布。

固定模型：[ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)。数据：[allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)。评分协议来源：[EleutherAI/lm-evaluation-harness v0.4.9](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9/lm_eval/tasks/arc)。本报告中的本地实验数值由公开结果文件生成。
'''
    (ROOT/'REPORT_ZH.md').write_text(text,encoding='utf8')
    save(options.receipt,dict(status='complete',candidate=cid,report='REPORT_ZH.md',
        gpu_worker_hours=budget['used_seconds']/3600,primary_plain_verdict=verdict(plain),primary_combo_verdict=verdict(fixed)))
    print('REPORT_COMPLETE',cid,flush=True)


if __name__=='__main__':main()
