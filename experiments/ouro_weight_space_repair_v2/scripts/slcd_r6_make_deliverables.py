"""Render evidence from immutable results. No inference, tuning, or selection.

--arc-plots permits an explicitly ARC-only figure preview during external jobs.
The final report requires both complete external tasks and independent audit.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def write(path,text):
    (ROOT/path).write_text(text,encoding='utf8')


def csvout(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT/path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def table(rows,columns):
    def display(value):return f'{value:.4f}' if isinstance(value,float) else str(value)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(display(row.get(c,'')) for c in columns)+' |' for row in rows)+'\n'


METHODS=['native4','fixed','adaptive','incumbent_A1','r6_prior_projection']
LABELS=['Native K4','Fixed','Adaptive','A1','R6']


def controls(result):
    return [row for row in result['results'] if row['baseline']=='fixed']


def method_name(name):
    return 'incumbent_A1' if name=='A1_g1p0' else name


def plots(tasks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.labelsize':13,'axes.titlesize':14,'xtick.labelsize':12,'ytick.labelsize':12})
    colors=['#AAB4C0','#5477A0','#7F92A7','#8A7DA8','#2D8C7F']
    fig,axes=plt.subplots(1,len(tasks),figsize=(max(6,4.8*len(tasks)),4.5),squeeze=False,layout='constrained')
    for ax,(title,result) in zip(axes[0],tasks):
        values={method_name(r['method']):r for r in controls(result)}
        bars=ax.bar(np.arange(5),[100*values[m]['accuracy'] for m in METHODS],color=colors,width=.7)
        for m,b in zip(METHODS,bars):
            r=values[m];ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.2,f"{100*r['accuracy']:.2f}\n({r['correct']})",ha='center',va='bottom',fontsize=11)
        ax.set(xticks=np.arange(5),xticklabels=LABELS,ylim=(0,100),title=f"{title} (n={result['n']:,})",ylabel='Accuracy (%)')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        if title=='ARC-Challenge':
            ax.axhline(62.5,color='#B46A53',linestyle=':',lw=1)
            ax.text(.02,.96,'25-shot; target 62.5%',transform=ax.transAxes,fontsize=11,va='top')
        else:ax.text(.02,.96,'0-shot; frozen rule',transform=ax.transAxes,fontsize=11,va='top')
    for suffix in ['pdf','png']:fig.savefig(ROOT/f'results/slcd_accuracy_comparison.{suffix}',dpi=200)
    plt.close(fig)
    comparisons=[]
    for title,result in tasks:
        records=result.get('paired_comparisons',result['results'])
        for baseline in ['fixed','incumbent_A1']:
            x=next(r for r in records if method_name(r['method'])=='r6_prior_projection' and method_name(r['baseline'])==baseline)
            comparisons.append((title+' vs '+('Fixed' if baseline=='fixed' else 'A1'),x))
    fig,(left,right)=plt.subplots(1,2,figsize=(13,max(4.3,.65*len(comparisons)+1.6)),layout='constrained',gridspec_kw={'width_ratios':[1,1.2]})
    y=np.arange(len(comparisons));width=.32
    left.barh(y-width/2,[x['corrections'] for _,x in comparisons],height=width,color='#2D8C7F',label='Rescues')
    left.barh(y+width/2,[x['regressions'] for _,x in comparisons],height=width,color='#B46A53',label='Damage')
    for i,(_,x) in enumerate(comparisons):
        left.text(x['corrections']+.3,i-width/2,str(x['corrections']),va='center',fontsize=11)
        left.text(x['regressions']+.3,i+width/2,str(x['regressions']),va='center',fontsize=11)
        delta=x['delta_pp'];low=x['paired_bootstrap_low_pp'];high=x['paired_bootstrap_high_pp']
        right.errorbar(delta,i,xerr=[[delta-low],[high-delta]],fmt='o',color='#5477A0',capsize=4)
        right.annotate(f"{delta:+.2f} pp; p={x['exact_mcnemar_p']:.3g}",(delta,i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
    left.set(yticks=y,yticklabels=[s for s,_ in comparisons],xlabel='Questions',title='Frozen R6 paired transitions')
    left.set_xlim(0,max(max(r['corrections'],r['regressions']) for _,r in comparisons)*1.18+2)
    left.legend(frameon=False,loc='lower right',fontsize=11)
    right.set(yticks=y,yticklabels=[],xlabel='Accuracy difference (percentage points)',title='Paired bootstrap 95% intervals')
    right.axvline(0,color='#8A8A8A',ls='--',lw=1)
    for ax in [left,right]:ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    for suffix in ['pdf','png']:fig.savefig(ROOT/f'results/slcd_rescue_regression_analysis.{suffix}',dpi=200)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arc-plots',action='store_true');parser.add_argument('--public-records',action='store_true');args=parser.parse_args()
    arc=read('results/slcd_R6_ARC_COMPLETE.json');acceptance=read('results/slcd_R6_TEST_ACCEPTANCE.json')
    assert arc['n']==1172 and arc['correct']==736 and acceptance['status']=='passed'
    assert sha('configs/slcd_R6_FINAL_LOCK.json')==arc['final_lock_sha256']==acceptance['final_lock_sha256']
    tasks=[('ARC-Challenge',arc)]
    if args.arc_plots:
        plots(tasks);print('ARC_ONLY_FIGURES_RENDERED');return
    chain=read('results/slcd_R6_EXTERNAL_CHAIN_COMPLETE.json')
    for task,title,n in [('hellaswag','HellaSwag',10042),('openbookqa','OpenBookQA',500)]:
        assert chain['tasks'][task]['status']=='complete'
        result=read(f'results/slcd_r6_{task}_COMPLETE.json')
        assert result['status']=='complete' and result['n']==n
        assert result['final_lock_sha256']==arc['final_lock_sha256']
        if args.public_records:
            public=read('results/slcd_R6_PUBLIC_NUMERIC_ACCEPTANCE.json')
            export=read('results/slcd_R6_PUBLIC_NUMERIC_EXPORT.json')
            assert public['status']=='passed' and public['export_receipt_sha256']==sha('results/slcd_R6_PUBLIC_NUMERIC_EXPORT.json')
            assert public['tasks'][task]['source_score_file_sha256']==export['tasks'][task]['source_score_file_sha256']==result['score_file_sha256']
            for p,meta in export['tasks'][task]['artifacts'].items():assert sha(p)==meta['sha256']
        else:assert sha(result['score_file'])==result['score_file_sha256']
        for p,value in result['artifact_sha256'].items():assert sha(p)==value
        tasks.append((title,result))
    audit=read('results/slcd_R6_EXTERNAL_ACCEPTANCE.json');assert audit['status']=='passed'
    manifest_audit=read('results/slcd_R6_MANIFEST_AUDIT.json');assert manifest_audit['status']=='passed'
    assert manifest_audit['external_lock_sha256']==sha('configs/slcd_R6_EXTERNAL_LOCK.json')
    assert all(manifest_audit['tasks'][task]['exact_full_row_equality'] for task in ['hellaswag','openbookqa'])
    budget=read('logs/slcd_gpu_budget.json');assert budget['used_seconds']<=21600
    assert all(r['status'] in ['complete','failed'] for r in budget['runs'])
    assert abs(sum(r['gpu_worker_seconds'] for r in budget['runs'])-budget['used_seconds'])<1e-6
    costs=[dict(job=r['name'],status=r['status'],actual_worker_seconds=r['gpu_worker_seconds'],actual_worker_hours=r['gpu_worker_seconds']/3600,exit_code=r.get('exit_code'),limit_seconds=r['limit_seconds']) for r in budget['runs']]
    csvout('results/slcd_R6_compute_costs.csv',costs)
    combined=[]
    for number in range(2,7):
        d=read(f'results/slcd_R{number}_DEVELOPMENT_COMPLETE.json')
        combined.extend(dict(round=f'R{number}',**x) for x in d['summary'])
    csvout('results/slcd_R6_all_followup_development.csv',combined)
    summary=[]
    for title,result in tasks:
        for x in controls(result):summary.append(dict(benchmark=title,method=method_name(x['method']),correct=x['correct'],n=x['n'],accuracy_percent=100*x['accuracy'],net_vs_fixed=x['net_correct']))
    csvout('results/slcd_R6_all_benchmark_results.csv',summary)
    plots(tasks)
    primary_fixed=next(r for r in arc['results'] if r['method']==arc['primary'] and r['baseline']=='fixed')
    primary_a1=next(r for r in arc['results'] if r['method']==arc['primary'] and r['baseline']=='incumbent_A1')
    dev=read('results/slcd_R6_DEVELOPMENT_COMPLETE.json');case=read('results/slcd_R6_CASE_AUDIT.json');overlap=read('results/slcd_R6_OVERLAP_SENSITIVITY.json')
    ext=[]
    for title,result in tasks[1:]:
        for base in ['fixed','A1_g1p0']:
            r=next(x for x in result['paired_comparisons'] if x['method']=='r6_prior_projection' and x['baseline']==base)
            ext.append(dict(benchmark=title,baseline=base,correct=f"{r['correct']}/{r['n']}",accuracy=f"{100*r['accuracy']:.3f}%",R=r['corrections'],D=r['regressions'],net=r['net_correct'],delta_pp=f"{r['delta_pp']:+.3f}",CI=f"[{r['paired_bootstrap_low_pp']:.3f}, {r['paired_bootstrap_high_pp']:.3f}]",p=f"{r['exact_mcnemar_p']:.5g}"))
    positive=sum(x['net']>0 for x in ext if x['baseline']=='fixed')
    generalization=('两个外部任务对 Fixed 的点估计均为正，但是否可靠须结合下表区间；不能据此宣称普遍优势。' if positive==2 else '外部任务未呈现一致的对 Fixed 正向迁移；ARC 达标与跨任务通用性是两个结论，不能合并。')
    followup=[]
    for number in range(2,7):
        d=read(f'results/slcd_R{number}_DEVELOPMENT_COMPLETE.json');candidates=[x for x in d['summary'] if x['subset']=='all' and x['baseline']=='incumbent_A1' and x['method'].startswith('r'+str(number)+'_')]
        best=max(candidates,key=lambda x:x['correct'])
        followup.append(dict(round=f'R{number}',best=best['method'],correct=best['correct'],net=best['net_correct'],test='仅选中投影：736' if number==6 else '未测试'))
    overlaps=[dict(scope=x['scope'],n=x['n'],correct=x['correct'],net_fixed=x['net_correct']) for x in overlap['primary_sensitivity'] if x['baseline']=='fixed']
    text=f'''# 原始 Ouro 权重的训练免费 LoopCD 改进：最终报告

冻结主方法 **R6 prior projection** 在完整 ARC-Challenge 上达到 **736/1172（62.7986%）**，超过用户要求的 62.5%（至少 733 题）。相对 Fixed 717 净增 19 题；相对已有 A1 726 净增 10 题。没有训练或编辑权重。全部工作保留在既有 V2 文件夹。

## Q1. 原始 LoopCD 为什么产生有害改答？

完整 ARC 上，Fixed 相对 Native K4 改答 95 题：纠错 41、损害 26、错到错 28；净增 15。它把早期到 K4 的 token-logit 差继续外推，这个方向同时包含有用和有害的排序位移。训练门控、几何投影和能量约束均被实际检验；单纯降低变化大小并未稳定解决问题。分数分解不能证明隐藏状态缺失、事实知识来源或某个神经元机制。

## Q2. 实现了哪些方法？

初始五轮注册 47 种含控制的配置：A1 完整候选分数晚期反转；A2 token-logit 晚期反转后重算完整 log-softmax；centered/raw 正交和平行引导；候选一致性/分差门控；最小混合；系数邻域和参考循环。用户提高目标后追加 R2–R6：Fisher late-residual 投影、匹配范数/余弦对照、分数复合、Fisher-RMS、精确 categorical-KL 约束、答案域先验以及四种部分/自适应先验修正。全代码与被否定分支保留。

原始 Fixed 在每个目标 token 上使用 z4+0.5(z4-z1)，再做全词表 log-softmax，完整累加候选似然；方法背景见 [Decoding Looped Transformers Better for (Almost) Free](https://arxiv.org/html/2610.02185v1)。A1 的候选级修正不同于在 token logits 上做相同形式的线性算术；二者的 log-partition 项通常依赖候选历史。所有归一化均只除一次相同答案字符数，不按 token 数，不截断候选。

## Q3. 自主迭代发现了什么？

初始正交/门控在小子集上上升，却未通过新增 517 题确认训练子集。R2/R4 的方向或信赖域操作没有超过已有 A1；R3 最好仅多1题，未获追加测试；R5 直接全强度减先验没有通过 +7 训练门槛。R6 改用当前候选集合上、有界且不用标签的投影系数，才获得较一致的训练净纠错。

{table(followup,['round','best','correct','net','test'])}
训练比较的 A1 分母均为 1093，正确 690。R2–R5 没有追加测试集候选评估；R3 最好只多 1 题，也未消耗全量测试。R1 五轮加用户目标扩展后的五轮共十轮，不能把整个历史写成只有五轮。推理公式不使用 gold、样本频率或答案出现次数。

## Q4. 开发选择与最终公式是什么？

原始 train1119 排除全部固定25示例并去重后为1093；划分384/192/517。R1 只在前三个晋级方法上做一次官方 validation299，A1=173、Fixed=168、Adaptive=172。后续 R2–R6 使用已暴露的同一1093训练样本和五个固定切片，没有新的独立验证集。

R6 projection 训练705，对 A1690 为22纠错/7损害；三子集净 +6/+1/+8；五切片净 +2/+3/+2/+4/+4。half=698、matched magnitude=692、quarter=690。通过预先规定的净增至少7题且至少3/5切片非负门槛后，只冻结 projection。五切片是反复使用训练数据的描述性稳健性检查，不是独立交叉验证。

令 S 为完整候选 LL/字符数，F 为 token-logit Fixed 的同口径分数：

```text
A = F + (S4 - S5)
g = F - S4
e = F_prior - S4_prior
gc = g - mean(g); ec = e - mean(e)
alpha = clip(dot(gc, ec) / dot(ec, ec), 0, 1)
        (dot(ec, ec) == 0 时 alpha = 0)
score = A - alpha * e
```

先验上下文是字面量 `Answer:`，保留完整候选续写，不保留题干或25示例。该先验估计文本/格式偏好，不能称为已识别的示例偏置。alpha 依赖当前候选集合；新增或删除选项可能改变排序。clip 后也不保证严格正交。分数是排名分数，不是校准概率。

最终锁 SHA256 `{arc['final_lock_sha256']}`，在测试前公开于 GitHub commit `496602d2416f8114c5227720235c26e5c14e488d`。根目录 `FINAL_METHOD_LOCK.json` 仍是历史 R1 锁；当前主方法使用 `configs/slcd_R6_FINAL_LOCK.json`，不得混淆。

## Q5. 完整 ARC 是否达到目标？

{table([x for x in summary if x['benchmark']=='ARC-Challenge'],['method','correct','n','accuracy_percent','net_vs_fixed'])}
模型 ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`，原始 BF16，GB10，seed2021，lm-eval0.4.9 acc_norm；同一组固定25示例，完整1172题。原始 safetensors SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`。

所有1172原始上下文分数在完整输入/模型/协议兼容核验后复用；R6 此次上下文 GPU 重放覆盖为0。新增先验完整覆盖1172行、4687候选，其中4408次新推理、279缓存命中。不能说此次重新对1172个主上下文做了 GPU 评估。

## Q6. 纠错、新错误和失败案例是什么？

相对 Fixed：35纠错、16损害、净+19。相对 A1：23纠错、13损害、12错到错、1124不改答。alpha=0/interior/1 分别覆盖419/562/191题；interior纠错17损害11，alpha=1纠错6损害2。419题严格回退为 A1。均为冻结后的描述性分析，没有据此改系数。

35个对Fixed的纠错中，10个恢复Native K4原本正确但被Fixed改错的答案，25个修复Native K4与Fixed都错的题。16个损害中，6个撤销Fixed的有益纠错，10个把Native K4与Fixed都正确的题改错。因此本数据上的净增19来自Native K4原错子集（25-6）；原正确子集上恢复与新增损害相抵（10-10）。这仍是结果分类，不是神经因果机制。

源顺序第一个纠错案例是行星自转加快：A1 选“重力增强”，R6 改为“行星日变短”，gold margin 从 -0.053662 变为 +0.004431。第一个损害案例是实验改变车速：A1 正确选自变量，R6 改成因变量，margin 从 +0.015070 变为 -0.010395。全48改答案例及全部4687候选的 g/e/alpha/修正项保存在 [案例报告](slcd_R6_FAILURE_CASES.md)、[JSONL](slcd_r6_arc_changed_cases.jsonl)、[候选分数](slcd_r6_arc_candidate_scores.csv)。这些例子展示了排序翻转，并未识别隐藏状态因果链。

## Q7. 统计与计算成本是否支持结论？

R6 vs Fixed：+{primary_fixed['delta_pp']:.4f}pp，配对50,000次题级 bootstrap95%CI [{primary_fixed['paired_bootstrap_low_pp']:.4f}, {primary_fixed['paired_bootstrap_high_pp']:.4f}]pp，双侧 exact McNemar p={primary_fixed['exact_mcnemar_p']:.7f}。R6 vs A1：+{primary_a1['delta_pp']:.4f}pp，CI [{primary_a1['paired_bootstrap_low_pp']:.4f}, {primary_a1['paired_bootstrap_high_pp']:.4f}]pp，p={primary_a1['exact_mcnemar_p']:.7f}。因此“达到用户整数目标”成立，“可靠超过 A1”尚不成立。

本轮额外 worker 累计 {budget['used_seconds']:.3f} 秒（{budget['used_seconds']/3600:.4f}小时），未超过6小时，包含失败/中止尝试；详见 [逐次成本](slcd_R6_compute_costs.csv)。worker 时间包含其 CPU/模型加载部分，不是 CUDA event 纯 GPU 时间；旧 V1/V2 缓存的生成成本不计入本轮新增预算。连接恢复未重置账本。

数学逻辑深度为主上下文K5、先验K4；需要的词表头为主1/4/5、先验1/4。实际收集统一运行共享K6轨迹，外部主上下文/先验长度不同。没有独立部署路径测速，不能声称延迟、FLOP或token吞吐收益。

## Q8. 是否跨任务泛化？

{table(ext,['benchmark','baseline','correct','accuracy','R','D','net','delta_pp','CI','p'])}
{generalization}

两个任务均为全量官方 split：HellaSwag validation10042，OpenBookQA main test500；官方 lm-eval0.4.9 零样本 prompt，acc_norm，同一冻结公式，无重调。A1 是固定历史对照。HellaSwag 复用1956行原始主分数，重放其源顺序首/中/末三题的八种readout完全一致；其余8086主上下文为新推理，不能把三题parity写成1956题独立重放。OpenBookQA500主上下文均为新推理。先验使用经授权 ARC 缓存或原始模型新推理，无独立历史 prior baseline。

另一项独立CPU检查从固定raw parquet与官方task定义重新渲染全部10542行，prompt、候选顺序、gold、请求边界和字符分母逐行完全相等，见 `slcd_R6_MANIFEST_AUDIT.json`。完整候选在各上下文分别按官方 encode_pair 边界编码；连续文本相同、字符分母相同，target token IDs 可能因上下文边界不同而不同，不强制裁剪成相同 token 段。外部任务与 ARC25-shot 的绝对准确率不直接比较。配对统计是描述性结果，未声称多重比较校正后的全局显著性。

## Q9. 是否继续，以及有哪些研究价值？

ARC 的62.5%目标已完成。值得保留的是“晚期反转 + 部分答案先验校正”的可复核排序干预，以及几何/能量控制失败的完整证据。下一阶段是否投入应由外部完整结果和新、未反复使用的数据决定；不能凭训练切片或一次已暴露测试集的达标就宣称通用改进。当前不启动新的大规模训练、循环结构或隐藏状态因果推断。

当前 R6 是针对给定候选集合的完整答案排名规则。它尚未实现或验证自由生成版本，不能将多项选择题上的提高直接推广为 GSM8K 自回归推理改善。候选集合变化对投影系数的影响是后续可证伪的问题，本轮没有据测试结果设计新的候选集合或再筛方法。

投影和校准均有明确先例：[Adaptive Projected Guidance](https://arxiv.org/abs/2410.02416)、[PMI_DC / Surface Form Competition](https://aclanthology.org/2021.emnlp-main.564/)、[Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html)。本轮是对 recurrent guidance 的具体适配，未证明首创、优于80%可比方法或揭示新神经机制。R2–R5的其他主文献边界见 [相关工作核验](../astra/slcd_FOLLOWUP_RELATED_WORK.md)。

## 完整性、数据重叠与交付范围

ARC test、官方validation和 A1历史726均已暴露；R6开发虽未读取新逐题test反馈，也不是新的盲测。原训练指纹审计按不同问题文本计数；后验逐测试行审计发现 {overlap['training_overlap']} 行与训练重叠、{overlap['validation_overlap']} 行与validation重叠，合计 {overlap['union_overlap']} 行。主结果保留全部1172，不事后删题；附加敏感性如下，排除重叠后净纠错19仍保留，但这些子集也不能称盲测。

{table(overlaps,['scope','n','correct','net_fixed'])}
完整代码、冻结配置、逐候选分数、转移、成本、两份PDF/PNG和核验记录随分支交付。完整输入及原始JSONL/hidden-state缓存保留服务器；模型权重不复制。HellaSwag上游仓库访问现被屏蔽，公开包保留完整数值、输入哈希与固定HF来源，不重复分发原文或可逆token IDs；其余输入依各自来源许可与attribution处理，见 [数据说明](../data/slcd_EXTERNAL_SOURCE_NOTICE.md)。

独立CPU审计重算1172个预测、5860条转移、全部配对统计、4408先验cache条目、37496先验LL与4687 contextual target IDs，均一致。审计使用本地tokenizers0.22.2、服务器0.21.4，具体输入token IDs精确相等，不能概括成两版本普遍等价。外部完整核验范围见 `slcd_R6_EXTERNAL_ACCEPTANCE.json`；所有旧671文件size/mtime和原模型完整字节SHA另有最终保留审计。

复现和恢复说明见 [README](../slcd_README.md)。原始R1报告归档为 `slcd_R1_REPORT_ZH.md`，R1锁/数据不被改写成R6。图表由本脚本直接读取实际完整结果绘制。
'''
    write('results/slcd_REPORT_ZH.md',text)
    readme=f'''# Frozen R6 training-free LoopCD: 736/1172 (62.80%)

Original BF16 Ouro-1.4B, fixed25-shot ARC, no training or weight editing. Target733 reached. Against Fixed717: rescue35/damage16, net+19, +1.6212pp, exactMcNemar p=.01097. Against A1=726:23/13, net+10; p=.13250, not reliable at.05. ARC is a historically reused regression benchmark.

{table(summary,['benchmark','method','correct','n','accuracy_percent','net_vs_fixed'])}
All external tasks use official0-shot prompts and the exact frozen rule. No tuning after test.

- [Chinese report, nine questions](results/slcd_REPORT_ZH.md)
- [Current R6 final lock](configs/slcd_R6_FINAL_LOCK.json), [external lock](configs/slcd_R6_EXTERNAL_LOCK.json)
- [Frozen score implementation](scripts/slcd_r6_scores.py)
- [Full ARC table](results/slcd_r6_arc_full_test_results.csv), [candidate scores](results/slcd_r6_arc_candidate_scores.csv), [paired transitions](results/slcd_r6_paired_transitions.csv), [cases](results/slcd_R6_FAILURE_CASES.md)
- [HellaSwag full table](results/slcd_r6_hellaswag_results.csv), [OpenBookQA full table](results/slcd_r6_openbookqa_results.csv)
- [Development](results/slcd_R6_all_followup_development.csv), [initial ablations](results/slcd_ablation_results.csv), [all iterations](SELF_ITERATION_LOG.md), [costs](results/slcd_R6_compute_costs.csv)
- [Accuracy figure](results/slcd_accuracy_comparison.pdf), [paired analysis](results/slcd_rescue_regression_analysis.pdf)
- [ARC independent audit](astra/slcd_R6_FINAL_CLAIM_AUDIT.md), [external acceptance](results/slcd_R6_EXTERNAL_ACCEPTANCE.json), [overlap sensitivity](results/slcd_R6_OVERLAP_SENSITIVITY.json)
- [Release manifest](artifacts/slcd_R6_RELEASE_INDEX.json) and [record contents](artifacts/slcd_R6_RECORDS_CONTENTS.json)

Model revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1; safetensors SHA58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. Environment and input hashes are locked. Logical mainK5/priorK4, actual sharedK6 collection; no latency claim. Additional worker total {budget['used_seconds']/3600:.4f}h/6h, excluding historical cache generation.

## Restore and verify saved evidence

In this existing experiment directory, fetch every release part listed by the release manifest, then run `python -B scripts/slcd_r6_restore_records.py --verify-only`. The manifest specifies whole archive/part SHA256 and individual record hashes. Omit `--verify-only` to restore absent records; different existing files are refused. Model/env/hidden-state caches and HellaSwag original text are not distributed. HellaSwag numeric scores and exact input hashes are preserved; obtain original data from the pinned source receipts to reconstruct prompts. Full originals remain on the server.

CPU ARC reproduction from restored records: `python -B scripts/slcd_r6_case_audit.py`. This computes only the frozen selected rule, without a model. `python -B scripts/slcd_r6_overlap_sensitivity.py` runs explicitly posthoc sensitivity. External numeric reproduction: `python -B scripts/slcd_r6_public_scores.py verify`, which reconstructs all five methods from exact raw LL and checks paired statistics and every public candidate CSV row. Then rebuild the final report/figures with `python -B scripts/slcd_r6_make_deliverables.py --public-records`. Original full-input server audits require raw inputs not duplicated publicly.

## Original-server inference and full-input audit

Use the unchanged helper/scorer code on this branch and the pinned model/environment in `env` and `models/Ouro-1.4B`. Keep all caches/temp files inside this existing folder. Every GPU job must use `scripts/slcd_supervise.py`, preserving its21600s ledger. `scripts/slcd_r6_prior_worker.py --split test` requires the frozen R6 final lock; `scripts/slcd_r6_final_chain.py` guards transitive hashes before and after evaluating ARC. The prior cache uses exact prefix/continuation/token IDs/model/source keys, never gold or frequencies.

After the complete ARC acceptance gate, external preprocessing uses `scripts/slcd_r6_prepare_external.py` and fixed raw parquets/source receipts. The joint worker is `scripts/slcd_r6_external_worker.py --task hellaswag` (or openbookqa); controller `scripts/slcd_r6_external_chain.py` runs HellaSwag then OpenBookQA. These exact commands are for the original server and require its immutable original main-score authority, raw inputs and caches, including the 1956-row old HellaSwag partial file. The public numeric package supports exact score/statistic reconstruction; it is not by itself the complete original-server runtime snapshot. Source-text-containing authorities remain on the server and are not fabricated or relabeled to bypass their hash locks. Existing completion files validate/reuse immutable evidence; a changed source, model, input, package, or cache raises an error. Inspect failed/stale supervisors before any retry. Re-running on the completed server does not authorize resetting budgets or replacing the final method.

The root `FINAL_METHOD_LOCK.json`, original `slcd_*` R1 test tables/receipts, and `slcd_R1_REPORT_ZH.md` are historical R1 evidence. The current R6 lock is explicitly under configs. Do not rerun method selection or substitute R1 metadata for R6. Original V2 README/REPORT and model weights remain preserved.
'''
    write('slcd_README.md',readme)
    status=readme.split('## Restore and verify saved evidence')[0]
    write('slcd_FOLLOWUP_STATUS.md',status+'\nFinal deliverables complete; see the current Chinese report and release manifest.\n')
    output=dict(status='rendered_from_complete_results',primary='r6_prior_projection',arc_correct=736,task_sizes={title:r['n'] for title,r in tasks},
                worker_seconds=budget['used_seconds'],source_sha256=sha('scripts/slcd_r6_make_deliverables.py'),
                report_sha256=sha('results/slcd_REPORT_ZH.md'),readme_sha256=sha('slcd_README.md'),
                figure_sha256={p:sha(p) for p in ['results/slcd_accuracy_comparison.pdf','results/slcd_rescue_regression_analysis.pdf']})
    write('results/slcd_R6_DELIVERABLES.json',json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
