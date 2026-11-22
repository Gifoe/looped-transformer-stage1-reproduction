"""Render the functional readout, feedback and external confirmation report."""
import argparse
import json
from pathlib import Path

NAMES={'native':'Native','previous_completion':'历史 Completion','previous_shuffled':'历史打乱教师',
       'scoreonly':'Score-only','dual':'Dual-objective','gate':'Utility Gate','confidence':'Confidence Gate',
       'teacher':'同题轮间写入监督','shuffled':'等范数打乱监督','functional':'功能分布监督',
       'feedback_only':'Feedback-only','readout_plus_feedback':'Readout + Feedback',
       'feedback_rotation':'等范数旋转反馈','feedback_state_distillation':'状态蒸馏反馈',
       'feedback_previous_direction':'旧方向反馈','feedback_shuffled_teacher':'旧打乱教师反馈'}
DEPTHS=(1,2,3,4)


def load(root,name):
    return json.loads((root/f'{name}.json').read_text(encoding='utf-8'))


def ci(obj):
    l,h=obj['ci95_pp']
    return f"{obj['delta_pp']:+.2f} [{l:+.2f}, {h:+.2f}]"


def profile(tab,key):
    return [tab['methods'][f'{key}_k{k}'] for k in DEPTHS]


def counts(tab,key):
    return '/'.join(str(r['correct']) for r in profile(tab,key))


def render(root,prior,review):
    checked=load(root,'VERIFICATION')
    s=load(root,'METRICS'); arc=s['arc']['metrics']; ext=s['openbookqa']['metrics']
    lock=load(root,'READOUT_FREEZE')['config']; fit=load(root,'feedback/FEEDBACK_FIT'); fb=load(root,'feedback/FREEZE')
    calibr=load(root,'CALIBRATION_FREEZE')['chosen']
    time=load(root,'ARC_READOUT_COST'); bench=load(root,'feedback/TEST_COST')['cost']
    solo=load(root,'feedback/ISOLATED_FEEDBACK_COST')['cost']
    past=load(prior/'results','COST')
    v=load(root,'feedback/MECHANISM'); par=load(root,'feedback/GENERATION_DIAGNOSTIC')
    minutes,seconds=divmod(checked['gpu_seconds'],60)
    lines=['# Ouro-1.4B 的功能读出与循环反馈','',
           f'主模型冻结。经确认使用单张 RTX 4090 D，GPU 台账 {minutes} 分 {seconds} 秒。ARC-Challenge 1172 题和 OpenBookQA 500 题均完整运行。核验了 {checked["matched_candidate_scores_in_new_artifacts"]:,} 个新候选分数。',
           f'ARC 的 `acc` 从原生 {counts(arc["acc"],"native")} 道变为联合方法 {counts(arc["acc"],"readout_plus_feedback")} 道，`acc_norm` 从 {counts(arc["acc_norm"],"native")} 道变为 {counts(arc["acc_norm"],"readout_plus_feedback")} 道。主要机制证据分别来自旧末端读出方向交换和本轮针对真实后缀的写回实验。','',
           '## 协议与冻结','',
           '沿用冻结的 Ouro-1.4B Base、BF16/SDPA、完整候选内容、全词表归一化。`acc` 用候选 token 对数概率求和，`acc_norm` 除以候选字符长度。ARC Train 1119 训练，Validation 299 选型。ARC Test 已被历史研究使用，按统一锁定复评解释。官方 OpenBookQA main/Test 500 题作为本次训练和选型之外的迁移测试。所有读出和反馈配置及代码在 ARC Test 前冻结，同历史逐题分数按 ID、候选顺序、token 长度和字符长度核对。本轮复用 ARC 历史原生与 LoopCD 分数。 [缓存与原始读出方法](../ouro_semantic_amplification_arc_20261122/REPORT_ZH.md) · [ARC 历史协议](../ouro_arc_pareto_20261119/REPORT_ZH.md) · [OpenBookQA 数据](https://huggingface.co/datasets/allenai/openbookqa)',
           '', '## A　低秩方向改变了什么','',
           f'Score-only 从原始候选分数、字符及 token 数量构造共享特征。验证比较 3 个训练损失权重和 45 个解析候选，其中锁定类型为 `{calibr["kind"]}`，参数或标量数为 {calibr["parameter_count"]}。另在 Validation 拟合读出分数变化，再于归档 Test 测量组内中心化变化的解释率。词频来自 ARC Train 的候选词。',
           '', '| 深度 | 分数及长度 held-out R² | 加训练词频后 R² | Score-only 与旧 Completion 的分数变化相关性 |',
           '|---|---:|---:|---:|']
    for k,a in s['calibration_explained'].items():
        lines.append(f"| {k.upper()} | {a['score_length_r2']:.3f} | {a['plus_train_word_frequency_r2']:.3f} | {a['score_only_shift_correlation']:.3f} |")
    lines.extend(['','旧读出的 299 题同缓存干预中，K3 等几何随机方向比真实方向少答对 35 道，`acc_norm` 相差 11.71 个百分点，经 Holm 校正 p=.027。跨题交换和原生分数/长度匹配交换保留了绝大多数按题中心化的候选分数变化，跨深度与真实更新相关分别为 0.959–0.978 和 0.962–0.983。遮蔽两类分支输入并保持剂量时高度相关，尚未定位一个对旧读出收益必要的输入分支。验证同缓存原生和历史原生的正确数存在小幅数值差异，消融只对照同缓存。','',
                  '12 组预先手工核对语义与表面的反事实中，语义改变前后都答对的归一化计数由原生 0/1/3/5 变为旧读出 1/1/3/3。只重排评分元数据时，并列最高分集合在各臂四个深度均保持 12/12。含候选列表的题面重排可改变预测。证据支持可跨题复用的决策边界修正，还不足以在旧末端读出中识别题目特异的功能语义。',
                  '', '## B　双目标读出与效用门控','',
                  r'共享秩 8 编码和输出。使用上一轮的四段支路写入、当前预归一化残差及历史差分。$h^{\prime}=h+d[g(x)w+u(x)]$，其中 $u\perp w$ 且有范数上限。损失在原候选分数向量处求两项目标梯度范数，按其比值的平方根平衡尺度。保留项在训练标签正确且原生缩放分数间隔至少 0.40 的位置约束候选分布。',
                  '双目标在 Validation 比较三个原始损失权重和三个剂量。实测同题轮间写入、等范数打乱方向及下一轮正确时的候选分布蒸馏三种辅助目标沿用锁定权重和剂量。门控从 1,119 个训练题的两折 out-of-fold 干预构造四预算的 4,476 条效用标签，折内头从全新初始化开始。两种门控分别读取分数置信度或额外的分支和残差统计，输出完整写入或零写入。部署用当前原生候选分数获取门控输入，因而需要额外一次词表头。',
                  '', '| 方法 | 验证 acc K1–K4 | 验证 acc_norm K1–K4 |', '|---|---:|---:|'])
    for name,method in lock.items():
        x=method['validation']
        lines.append(f"| {NAMES[name]} | {'/'.join(map(str,x['acc']))} | {'/'.join(map(str,x['acc_norm']))} |")
    lines.extend(['',f"选中的共享原始排序损失权重为 {lock['dual']['weight']:.2f}，剂量 {lock['dual']['dose']:.2f}。",'',
                  '## C　真实循环写回','',
                  f'K2–K4 在倒数第二次共享栈结束的归一化前改写残差，下一轮读取修改后的真实状态。K1 在物理第 21 层后改写，继续执行最后三层。只编辑每个候选用于预测答案 token 的位置。因果检查在四档改变未来 token 后得到零早期输出差，零剂量对照也精确复现未编辑状态。共享反馈模块参数 {fit["trainable_parameters"]:,}，梯度穿过冻结后缀，训练 1,119 题，平均裁剪前梯度范数 {fit["history"][0]["mean_preclip_grad_norm"]:.3f}。验证锁定 `{fb["chosen"]}`。',
                  '', '| 深度 | 48 题全反馈 acc_norm | 两支路均钳回原生 | 只恢复 Attention | 只恢复 MLP | 全反馈相对双钳制的归一化金标 margin 差 [95% CI] |',
                  '|---|---:|---:|---:|---:|---|'])
    # The mechanism producer stores the four depth count dictionaries together.
    for k in DEPTHS:
        key=f'k{k}'
        group=v['counts'].get(key,v['counts'])
        values=[group[f'{arm}_{key}']['acc_norm'] for arm in ('feedback_full','feedback_direct_residual','feedback_restore_attention','feedback_restore_mlp')]
        c=next(x for x in v['paired_contrasts'] if x['depth']==k and x['contrast']=='computed_suffix_over_direct' and x['metric']=='acc_norm')
        l,h=c['margin_paired_bootstrap_95']
        lines.append('| '+' | '.join([f'K{k}',*map(str,values),f"{c['margin_difference']:+.3f} [{l:+.3f}, {h:+.3f}]"])+' |')
    lines.extend(['','双钳制保留直接残差，只将实际后缀各层的 Attention 与 MLP 写入替换为同题原生值。表中 margin 区间是预先定义对比的名义配对区间。整答配对在机制小样本上经过多重校正均未达显著，分支单独恢复只作描述。大样本准确率与小样本路径证据共同支持后续原生计算使用了更新，尚未证明该更新编码了可命名的中间语义。','','## D　完整 ARC 与外部迁移',''])
    requested=list(NAMES)
    for label,tab,n in (('ARC-Challenge',arc,1172),('OpenBookQA',ext,500)):
        lines.extend([f'### {label} （{n} 题）','',
                      '| 方法 | acc 正确数 K1/K2/K3/K4 | acc_norm 正确数 K1/K2/K3/K4 |', '|---|---:|---:|'])
        if label=='ARC-Challenge':
            for prefix,desc in (('loopcd_','历史最优 LoopCD'),):
                a=[]; b=[]
                for entry,kind in ((a,'acc'),(b,'acc_norm')):
                    for k in DEPTHS:
                        opts=[x['correct'] for key,x in tab[kind]['methods'].items() if key.startswith(prefix) and key.endswith(f'k{k}')]
                        entry.append(str(max(opts)) if opts else '—')
                lines.append(f"| {desc} | {'/'.join(a)} | {'/'.join(b)} |")
        for arm in requested:
            if f'{arm}_k1' in tab['acc']['methods']:
                lines.append(f"| {NAMES[arm]} | {counts(tab['acc'],arm)} | {counts(tab['acc_norm'],arm)} |")
        lines.extend(['', '| 指标与深度 | 联合对原生救回/退化 | 差值 pp [95% CI] | Holm p（八端点） | 联合对同深度最强历史方法差值及区间 |',
                      '|---|---:|---|---:|---|'])
        for m in ('acc','acc_norm'):
            for k,r in zip(DEPTHS,profile(tab[m],'readout_plus_feedback'),strict=True):
                a,b=r['vs_native'],r['vs_previous_best']
                suffix=f"{ci(b)} · `{tab[m]['previous_best'][f'k{k}']}`" if label=='ARC-Challenge' else '—'
                lines.append(f"| {m} K{k} | {a['rescued']}/{a['damaged']} | {ci(a)} | {a['holm8_p']:.3g} | {suffix} |")
        lines.append('')
    lines.extend(['同一尺度下 K1–K4 的逐题对照，包括 Gate 对无门控/Confidence-only、三种辅助监督与四类反馈对照，见 `METRICS.json`。所有损坏样本计数、分数 margin 与选择长度变化另存于 `FAILURES.json`。','',
                  '### 主对照与归因','', '| 对照 | acc 净差 pp K1/K2/K3/K4 | acc_norm 净差 pp K1/K2/K3/K4 |', '|---|---:|---:|'])
    for key in ('dual_vs_scoreonly','gate_vs_dual','gate_vs_confidence','teacher_vs_shuffled','functional_vs_dual',
                'functional_vs_shuffled','feedback_only_vs_feedback_rotation','feedback_only_vs_feedback_state_distillation',
                'readout_plus_feedback_vs_dual','readout_plus_feedback_vs_feedback_only'):
        both=[' / '.join(f"{x['delta_pp']:+.2f}" for x in arc[m]['contrasts'][key].values()) for m in ('acc','acc_norm')]
        lines.append(f"| `{key}` | {' | '.join(both)} |")
    lines.extend(['',f'OpenBookQA 与 ARC 训练或验证题干的规范化精确匹配为 {checked["openbookqa_exact_normalized_stem_overlap_with_arc_fit"]}/500。',
                  'OpenBookQA 的 `METRICS.json` 同时给出联合和反馈相对同协议 Score-only 的配对结果。',
                  '', '| ARC 跨预算联合对历史最强 | acc 差及 95% CI | acc_norm 差及 95% CI |', '|---|---|---|'])
    for key in arc['acc']['cross_budget']:
        data=[arc[m]['cross_budget'][key] for m in ('acc','acc_norm')]
        lines.append(f"| `{key}` | {ci(data[0])} `{data[0]['reference']}` | {ci(data[1])} `{data[1]['reference']}` |")
    lines.extend(['',f'自由生成诊断在 24 道固定 Validation 题上运行。选项显式列在题面，统一显示 A–E 标记，使用全词表贪心生成，最多六 token。K1–K4 联合方法解析正确数 '+
                  '/'.join(str(par['scores'][f'combo_k{k}']) for k in DEPTHS)+'，同提示原生为 '+
                  '/'.join(str(par['scores'][f'native_k{k}']) for k in DEPTHS)+'。联合方法的可解析标签数为 '+
                  '/'.join(str(par['valid'][f'combo_k{k}']) for k in DEPTHS)+'，仅反馈为 '+
                  '/'.join(str(par['valid'][f'feedback_k{k}']) for k in DEPTHS)+'。初跑提示要求字母但有一道数字标号题，原始记录另存。标签修正独立于已锁定的候选评分。','',
                  '## 实测计算成本','', '| 深度 | 双目标全 Test 秒数 | 效用门控全 Test 秒数 | 反馈与联合同时计分秒数 | 96 题独立联合中位批 ms/峰值 GiB | 历史 Completion 全 Test 秒数 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for k in DEPTHS:
        a=time[f'dual_k{k}'];b=time[f'gate_k{k}'];c=bench[f'k{k}'];d=solo[f'readout_plus_feedback_k{k}'];e=past[f'completion_k{k}']
        lines.append(f"| K{k} | {a['wall_s']:.2f} | {b['wall_s']:.2f} | {c['seconds_full_split']:.2f} | {d['median_batch_ms']:.2f}/{d['peak_bytes']/2**30:.2f} | {e['seconds_full_split']:.2f} |")
    lines.extend(['',f'各配置实际运行 K 次共享栈。联合模块有两个 {fit["trainable_parameters"]:,} 参数读出块，部署单输出仅用一次词表头，96 题的独立计时验证了这条路径。联合全量计时包含同一次主干上的两个词表头，以便同时测量反馈与联合准确率。前轮和本轮批处理口径不同，跨方法时延作为实测参考。Score-only 的完整四档 CPU 后处理为 {time["scoreonly"]["cpu_postprocessing_all_four_s"]:.3f} 秒，尚需原生候选分数。逐配置记录了模型调用、峰值、已执行层数和计时。','',
                  '## Q1–Q6',''])
    gate_line=[]
    for a in ('gate','confidence'):
        pct='/'.join(f"{100*time[f'{a}_k{k}']['active_fraction']:.1f}%" for k in DEPTHS)
        gate_line.append(f'{NAMES[a]} 的四档实际写入率为 {pct}，控制器参数量为 {time[f"{a}_k1"]["gate_parameter_count"]}。')
    rel='/'.join(f"{100*bench[f'k{k}']['relative_write_rms']:.1f}%" for k in DEPTHS)
    gate_line.append(f'反馈更新相对原生残差的 RMS 均值为 {rel}。低秩限制压缩了参数量，实际写入幅度可达残差的数成。')
    lines[-2:-2]=[''.join(gate_line),'']
    return '\n'.join(lines)+review


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('results','previous','review','output'):
        p.add_argument(n,type=Path)
    a=p.parse_args();a.output.write_text(render(a.results,a.previous,a.review.read_text(encoding='utf-8')),encoding='utf-8')
