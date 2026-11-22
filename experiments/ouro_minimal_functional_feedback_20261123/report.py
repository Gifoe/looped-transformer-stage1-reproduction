"""Audit all finished runs, apply the preregistered sufficiency rule and render the report."""
import argparse
import json
from pathlib import Path

from minimal import DEPTHS, dump, load_jsonl, sha


METRICS=('acc','acc_norm')
CASES={'ARC':(1172,'arc_validation',299),'OPENBOOKQA':(500,'openbookqa_test',500)}
NAMES={'native':'Native','feedback_only':'原六路反馈','readout_plus_feedback':'原联合方法',
       'linear8_t1':'线性 r8','nonlinear8_t1':'非线性 r8','history8_t1':'历史差分 r8',
       'adapter8_t1':'无幅度上限 Adapter r8','distill8_t1':'状态蒸馏 r8','constant_t1':'共享常量'}


def read(root,name):
    return json.loads((root/f'{name}.json').read_text(encoding='utf-8'))


def p(x):
    lo,hi=x['ci95_pp']; return f"{x['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def ktab(x,metric,method):
    arm=x['metrics'][metric]['methods']; return [arm[f'{method}_k{k}'] for k in DEPTHS]


def counts(x,metric,method):
    return [s['correct'] for s in ktab(x,metric,method)]


def flat(x):
    return '/'.join(map(str,x))


def verify(root,cfg,m):
    files={}
    for label,(n,_,_) in CASES.items():
        path=root/f'RESULTS_{label}.jsonl'; q=load_jsonl(path)
        assert len(q)==len({a['id'] for a in q})==m[label]['n']==n
        expected={f'{a}_k{k}' for a in cfg['arms'] for k in DEPTHS}
        assert all(expected<=x['scores'].keys() and set(x['predictions'])==set(METRICS) for x in q)
        files[path.name]=sha(path)
    path=root/'generation/GENERATION_RESULTS.jsonl'; x=load_jsonl(path)
    gener=read(root,'generation/GENERATION_SUMMARY')['runs']
    for _,(_,domain,n) in CASES.items():
        for method in ('native','original_feedback','original_joint',*cfg['generation_arms']):
            assert all(gener[f'{domain}__{method}__k{k}']['n']==n for k in DEPTHS)
    assert len(x)==sum(a['n'] for a in gener.values())
    files[path.name]=sha(path)
    assert len(load_jsonl(root/'CAUSAL_ANALYSIS/B_PREDICTIONS.jsonl'))==299
    assert len(load_jsonl(root/'CAUSAL_ANALYSIS/C_PREDICTIONS.jsonl'))==299
    ledger=(root/'GPU_LEDGER.tsv').read_text().splitlines()
    assert ledger and ledger[-1].startswith('END\t')
    stages=[]
    for start_line,end_line in zip(ledger[::2],ledger[1::2],strict=True):
        _, stage, start, remainder=start_line.split('\t')
        close, ended_stage, started_again, end, seconds, code=end_line.split('\t')
        assert close=='END' and (stage,start)==(ended_stage,started_again)
        assert int(end)-int(start)==int(seconds)
        stages.append(dict(phase=stage,elapsed_seconds=int(seconds),exit_code=int(code),
                           budget_before_seconds=int(remainder)))
    elapsed=sum(record['elapsed_seconds'] for record in stages)
    assert elapsed<=18000
    return dict(counts={k:v[0] for k,v in CASES.items()},generation_records=len(x),gpu_seconds=elapsed,
                gpu_stages=stages,hashes=files)


def levels(cfg,m,g):
    choice=cfg['best']; rule=cfg['sufficiency']
    record={}; compact=1-cfg['arms'][choice]['params']/116753
    for label,(n,name,gen_n) in CASES.items():
        now=[x for metric in METRICS for x in counts(m[label],metric,choice)[1:]]
        original=[x for metric in METRICS for x in counts(m[label],metric,'feedback_only')[1:]]
        native=[x for metric in METRICS for x in counts(m[label],metric,'native')[1:]]
        denom=sum(original)-sum(native)
        retain=(sum(now)-sum(native))/denom if denom>0 else None
        by_gen={}
        for metric in ('valid','correct'):
            current=sum(g[f'{name}__{choice}__k{k}'][metric] for k in (2,3,4))
            prior=sum(g[f'{name}__original_feedback__k{k}'][metric] for k in (2,3,4))
            by_gen[metric]=100*(current-prior)/(3*gen_n)
        record[label]=dict(gain_retention=retain,all_six_at_least_prior=all(a>=b for a,b in zip(now,original,strict=True)),
                           a_cell_improved=any(a>b for a,b in zip(now,original,strict=True)),generation_delta_pp=by_gen,
                           generation_maintained=min(by_gen.values())>=-rule['generation_noninferiority_pp'])
    maintained=all(a['generation_maintained'] for a in record.values())
    a=maintained and cfg['arms'][choice]['params']<=32768 and all(x['all_six_at_least_prior'] for x in record.values())
    b=(maintained and compact>=rule['parameter_reduction'] and all(x['gain_retention'] is not None and
         x['gain_retention']>=rule['retention'] for x in record.values()) and any(x['a_cell_improved'] for x in record.values()))
    return dict(selection=choice,parameter_reduction=compact,results=record,level_A=a,level_B=b,loopcd_extension_gate=a or b)


def render(root,out):
    cfg=read(root,'FREEZE'); m=read(root,'BENCHMARK'); costs=read(root,'COST'); v=read(root,'VALIDATION')
    g=read(root,'generation/GENERATION_SUMMARY')['runs']; g_cost=read(root,'generation/GENERATION_COST')['runs']
    spec=read(root,'SUBSPACE_ANALYSIS/FUNCTIONAL_GRADIENT_SPECTRUM'); a=read(root,'SUBSPACE_ANALYSIS/ARC_GRADIENT_ALIGNMENT')
    ext=read(root,'SUBSPACE_ANALYSIS/OPENBOOKQA_GRADIENT_ALIGNMENT'); ini=read(root,'INIT_DECISION')
    b=read(root,'CAUSAL_ANALYSIS/B_FACTORIAL'); c=read(root,'CAUSAL_ANALYSIS/C_POSITIONS')
    check=verify(root,cfg,m); verdict=levels(cfg,m,g)
    dump(root/'VERIFICATION.json',check); dump(root/'SUFFICIENCY.json',verdict)
    costs['_gpu_budget']=dict(limit_seconds=18000,used_seconds=check['gpu_seconds'],stages=check['gpu_stages'],
                              scope='Conservative serial GPU lease wall time including launch, CPU work and failed starts.')
    costs['_generation']={key:{field:value for field,value in timing.items() if field!='batches'}
                          for key,timing in g_cost.items()}
    dump(root/'COST.json',costs)
    best=cfg['best']; size=cfg['arms'][best]['params']; minutes,seconds=divmod(check['gpu_seconds'],60)
    levels_met='、'.join(k for k,v in [('A',verdict['level_A']),('B',verdict['level_B'])] if v) or '无'
    lines=['# Ouro 最简功能反馈','',f'在已授权的单张 RTX 4090 D 完成 ARC 1,172 题、OpenBookQA 500 题及两套全量自由生成，计入失败尝试累计 {minutes} 分 {seconds} 秒。候选基线来自同一冻结模型的历史逐题记录。',
           f'按 ARC Validation 两种候选指标的四档合计冻结 `{best}`，参数 {size:,}，较原六路反馈少 {100*verdict["parameter_reduction"]:.2f}%。达到的简化等级为 {levels_met}。',
           '', '## 方法与效用对照', '',
           '低秩模块只增加预归一化残差，原生最后一圈或后续三层在真实输入上执行。位置、训练集合、训练步数和损失见 [方法](METHOD.md)。用原生词表头在完整候选上累加完整词表归一的对数概率。`acc_norm` 除以字符数。K1–K4 均为实际完整通过的 Loop 数。模型选择只读 ARC Validation，ARC Test 在本研究体系内属于历史统一复评。OpenBookQA 不参与训练或选型。 [历史反馈](../ouro_functional_readout_feedback_20261122/REPORT_ZH.md)',
           '', '| Validation 配置 | 参数 / 可训练 | 剂量 | acc K1–K4 | acc_norm K1–K4 | 训练秒数 | 16 候选中位批毫秒 K1–K4 |', '|---|---:|---:|---:|---:|---:|---:|']
    for tag,row in v.items():
        f=row['fit']; vv=row['counts']
        times=row['options'][str(row['dose'])]['cost']
        median=flat([f"{times[f'k{k}']['median_batch_ms']:.2f}" for k in DEPTHS])
        lines.append(f"| `{tag}` | {f['total_parameters']:,}/{f['trainable_parameters']:,} | {row['dose']:.2f} | {flat(vv['acc'])} | {flat(vv['acc_norm'])} | {f['seconds']:.2f} | {median} |")
    initializer=root/'SUBSPACE_ANALYSIS/SUBSPACE_RANDOM_INIT_CONTROLS.json'
    if initializer.exists():
        lines.extend(['', '| r8 初始化对照 | 随机 U 的 acc / acc_norm | 功能梯度 U 的 acc / acc_norm |',
                      '|---|---|---|'])
        for name,row in json.loads(initializer.read_text(encoding='utf-8'))['comparisons'].items():
            raw=row['random_control']
            functional=row['gradient_initialized']
            lines.append(f"| `{name}` | {flat(raw['acc'])} · {flat(raw['acc_norm'])} | {flat(functional['acc'])} · {flat(functional['acc_norm'])} |")
    for label,(n,_,_) in CASES.items():
        actual=m[label]
        lines.extend(['',f'## {label} 完整候选评分（{n} 题）','',
                    '| 方法 | acc 正确数 K1–K4 | acc_norm 正确数 K1–K4 | 参数 |', '|---|---:|---:|---:|'])
        for tag in (*REFERENCE,*cfg['arms']):
            size=cfg['arms'][tag]['params'] if tag in cfg['arms'] else {'native':0,'feedback_only':116753,'readout_plus_feedback':233506}[tag]
            lines.append(f"| {NAMES.get(tag,tag)} | {flat(counts(actual,'acc',tag))} | {flat(counts(actual,'acc_norm',tag))} | {size:,} |")
        lines.extend(['','| 指标深度 | 选中最简对 Native 救回/退化 | 净差 pp [95% CI] | 对原反馈救回/退化 | 净差 pp [95% CI] | Holm p 对原反馈（八端点） |',
                      '|---|---:|---|---:|---|---:|'])
        for metric in METRICS:
            for k,res in zip(DEPTHS,ktab(actual,metric,best),strict=True):
                x,y=res['vs_native'],res['vs_feedback_only']
                lines.append(f"| {metric} K{k} | {x['rescued']}/{x['damaged']} | {p(x)} | {y['rescued']}/{y['damaged']} | {p(y)} | {y['p_holm']:.3g} |")
        r=verdict['results'][label]
        ratio='不可定义' if r['gain_retention'] is None else f"{100*r['gain_retention']:.1f}%"
        lines.extend(['',f'K2–K4 两指标合计相对原反馈的净增益保留率为 {ratio}。所有当前方法的分数及预测可由逐题文件重算。'])
        if label=='ARC':
            lines.extend(['', '| 指标深度 | 同档历史 LoopCD | LoopCD 正确数 | 最简对 LoopCD 救回/退化 | 净差 pp [95% CI] |',
                          '|---|---|---:|---:|---|'])
            for metric in METRICS:
                archive=actual['metrics'][metric]
                for k,new in zip(DEPTHS,ktab(actual,metric,best),strict=True):
                    name=archive['top_loopcd'][f'k{k}']
                    if name is not None:
                        pair=new['vs_top_loopcd']
                        lines.append(f"| {metric} K{k} | `{name}` | {archive['methods'][name]['correct']} | {pair['rescued']}/{pair['damaged']} | {p(pair)} |")
    lines.extend(['','## 全量真实贪心生成','',
                  '统一展示 A–E 标签并请求输出字母，原生与各模块使用相同题面和六 token 上限。无选项 token 过滤。逐题文件保留实际生成 token、文本与格式无效类别。',
                  '', '| 数据与方法 | 正确 K1–K4 | 有效 K1–K4 | 均值生成 token K1–K4 | 全量耗时秒 K1–K4 |', '|---|---:|---:|---:|---:|'])
    for label,(_,domain,n) in CASES.items():
        for tag in ('native','original_feedback','original_joint',*cfg['generation_arms']):
            keys=[f'{domain}__{tag}__k{k}' for k in DEPTHS]; rows=[g[x] for x in keys]
            lines.append(f"| {label} {tag}（{n}） | {flat([x['correct'] for x in rows])} | {flat([x['valid'] for x in rows])} | {flat([format(x['mean_tokens'],'.2f') for x in rows])} | {flat([format(g_cost[x]['wall_s'],'.1f') for x in keys])} |")
    lines.extend(['', '生成汇总记录严格单字母输出数、无效类型、停止原因和对 Native 的配对区间。计成本时保留各方法的实际生成长度，保留率门槛另行比较原六路反馈。'])
    generation_pairs=root/'generation/GENERATION_PAIRED.json'
    if generation_pairs.exists():
        paired=json.loads(generation_pairs.read_text(encoding='utf-8'))['splits']
        lines.extend(['', '| 数据 | 与原六路相比的 K2–K4 正确率 pp [95% CI] | 有效率 pp [95% CI] |',
                      '|---|---|---|'])
        for label,(_,domain,_) in CASES.items():
            x=paired[domain][f'{best}_vs_original_feedback']['pooled_k2_k4']
            lines.append(f"| {label} | {p(x['correct'])} | {p(x['valid'])} |")
    lines.extend(['','## 梯度谱与低秩投影','',
                  '中心化谱衡量题间变化，均值能量单独报告。谱是把一个向量加到一题所有候选预测位置的梯度族。每档 Train 320 题、Validation 299 题，OpenBookQA 192 题只在全部配置冻结后用于描述。 [数学定义](THEORY.md)',
                  '', '| 梯度训练组 | 均值能量 | 前 1、2、4、8、16 维的中心化比例 |', '|---|---:|---:|'])
    for part,row in spec['train']['shared'].items():
        values=flat([f"{100*row['centered']['top_energy_fraction'][str(r)]:.1f}%" for r in (1,2,4,8,16)])
        lines.append(f"| {part} | {100*row['mean_energy_fraction']:.1f}% | {values} |")
    lines.extend(['',f'联合中心化 r8 的 Train 梯度能量覆盖为 {100*ini["joint_centered_r8"]:.1f}%，初始化门槛为 25%。功能梯度基初始化实际{"执行" if ini["attempt"] else "跳过"}。',
                  '', '| 深度 | ARC 验证共用梯度基 / 学习 U / PCA / 随机 | 外部对应四种基的能量覆盖 |', '|---|---:|---:|'])
    for k in DEPTHS:
        rows=[]
        for source in (a,ext):
            q=source['gradient_projections'][f'shared/k{k}/r8']
            rows.append(flat([f"{100*q[x]['centered']['aggregate_energy_fraction']:.1f}%" for x in ('arc_joint_gradient_centered','learned_u','arc_hidden_joint_pca','random')]))
        lines.append(f'| K{k} | {rows[0]} | {rows[1]} |')
    projection=read(root,'SUBSPACE_ANALYSIS/SUBSPACE_INTERVENTION')['paired_results']
    lines.extend(['','| 深度 | 原学习方向 | 同档功能梯度投影 | 共用功能梯度投影 | 残差 PCA 投影 | 随机投影 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for k in DEPTHS:
        cells=[projection[f'{tag}/historic_native/k{k}/acc_norm']['correct']
               for tag in ('full','depth_gradient','joint_gradient','hidden_pca','random')]
        lines.append(f'| K{k} | '+' | '.join(map(str,cells))+' |')
    lines.extend(['','表中是 ARC Validation 299 题的 `acc_norm` 正确数。每个投影保持原学习更新的逐 token 范数，实际 BF16 误差、配对区间和主角度在 `SUBSPACE_ANALYSIS`。','',
                  '## 方向与后续计算','',
                  '差中差比较学习和旋转两方向在开启或钳制后续 Attention/MLP 时的效用，四格保留直接残差。配对样本为 ARC Validation 299 题。',
                  '', '| 深度 | acc 差中差 pp | acc_norm 差中差 pp | 归一化 margin 差中差 [Bonferroni 95% CI] |', '|---|---:|---:|---|'])
    for k in DEPTHS:
        subset={x['metric']:x for x in b['factorial'] if x['test']==f'k{k}/direction_x_computation'}
        z=subset['margin_norm']; lo,hi=z['bonferroni_bootstrap_95']; lines.append(f"| K{k} | {100*subset['acc']['difference']:+.2f} | {100*subset['acc_norm']['difference']:+.2f} | {z['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    lines.extend(['','| 固定 K4 的写入日程 | acc 正确数 | acc_norm 正确数 | acc_norm 相对 Native pp [95% CI] |',
                  '|---|---:|---:|---|'])
    for name,schedule in c['schedules'].items():
        if not schedule:
            continue
        tags={x['metric']:x for x in c['paired'] if x['test']==f'{name}_vs_native'}
        r=tags['acc_norm']
        d,lo,hi=(100*a for a in (r['difference'],*r['paired_bootstrap_95']))
        lines.append(f"| `{name}` | {round(299*tags['acc']['scores'][name])} | {round(299*r['scores'][name])} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines.extend(['','跨题交换、常量方向及蒸馏的等范数对照见 `CAUSAL_ANALYSIS/B_FACTORIAL.json`。固定 K4 的提前写回、重复写回和剩余长度两折条件均保持四次原生通过。各位置的实际扰动 RMS 和轨迹见 `CAUSAL_ANALYSIS/C_POSITIONS.json`。','',
                  '## 成本与扩展门槛','', '| 数据与最简模型 | 完整候选评分秒数 K1–K4 | 16 候选中位批毫秒 K1–K4 | 峰值 GiB K1–K4 |', '|---|---:|---:|---:|'])
    for label in CASES:
        x=[costs[label][best]['depths'][f'k{k}'] for k in DEPTHS]
        lines.append(f"| {label} | {flat([format(d['seconds'],'.2f') for d in x])} | {flat([format(d['median_batch_ms'],'.2f') for d in x])} | {flat([format(d['peak_bytes']/2**30,'.2f') for d in x])} |")
    lines.extend(['','同口径的新方法逐档计算调用和实际延迟在 `COST.json`。历史候选时间来自已归档的对应运行。严格时延比较限于同一测量运行。 [历史 Native 与 LoopCD](../ouro_arc_pareto_20261119/results/standalone_cost.json) [历史六路与联合反馈](../ouro_functional_readout_feedback_20261122/results/feedback/TEST_COST.json)',
                  f'冻结门槛要求同一模型保留两数据集大部分候选收益与真实生成。LoopCD 扩展门槛为 {"通过" if verdict["loopcd_extension_gate"] else "未通过"}。',
                  '', '## 最终问题复核', ''])
    if (root.parent/'SCIENTIFIC_REVIEW_ZH.md').exists():
        lines.append((root.parent/'SCIENTIFIC_REVIEW_ZH.md').read_text(encoding='utf-8'))
    out.write_text('\n'.join(lines)+'\n',encoding='utf-8')


REFERENCE=('native','feedback_only','readout_plus_feedback')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    render(args.root,args.out)
