"""Render only accepted complete evidence; never selects a test winner."""
import csv,json
import numpy as np
from r11_core import ROOT,read,sha,save,rows
from r11_evaluate import records,csvout
from slcd_r6_scores import _inputs,projection_parameters


def table(data,keys):
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(str(r.get(k,'')) for k in keys)+' |' for r in data])


def main():
    acceptance=read('results/r11_EXPERIMENT_ACCEPTANCE.json');assert acceptance['status']=='passed_original_server_experiment_integrity'
    arc=read('results/r11_ARC_COMPLETE.json');primary=arc['primary'];data=records('test');m=arc['metrics'];comp={r['reference']:r for r in arc['comparisons']}
    profile=read('results/r11_DEPLOYMENT_PROFILE.json');docs={r['item']['id']:r for r in rows('test')};cases=[];strata=[]
    for row in data:
        if row['predictions'][primary]==row['predictions']['M0_R6']:continue
        source=docs[row['id']];item=source['item'];a,g,e=_inputs(source['original'],source['prior']);oldalpha=projection_parameters(g,e)[0]
        diag=row['diagnostics'][primary];newalpha=diag.get('alpha',oldalpha);p=np.asarray(diag.get('removed_component',[0.]*len(a)));old=np.asarray(row['baseline_scores']['M0_R6']);new=np.asarray(row['method_scores'][primary])
        coef=(oldalpha-newalpha)*(e-.5*p);protected=.5*(oldalpha+newalpha)*p;residual=new-old-coef-protected;assert np.max(np.abs(residual))<1e-12
        gold=row['gold'];cases.append(dict(id=row['id'],transition=('Correct' if row['correct']['M0_R6'] else 'Wrong')+'->'+('Correct' if row['correct'][primary] else 'Wrong'),
            question=item['question'],choices=item['choices'],gold=gold,R6_prediction=row['predictions']['M0_R6'],new_prediction=row['predictions'][primary],
            R6_scores=old.tolist(),new_scores=new.tolist(),old_alpha=oldalpha,new_alpha=newalpha,reference_consensus=diag['protection_consensus'],
            averaged_coefficient_component=coef.tolist(),averaged_protection_component=protected.tolist(),explicit_FP64_residual=residual.tolist(),
            R6_unique_rescue_vs_Fixed=row['correct']['M0_R6'] and not row['correct']['fixed'],
            interpretation='Two-order averaged algebra with interaction, not unique semantic or hidden-state causation'))
    with (ROOT/'results/r11_failure_cases.jsonl').open('w',encoding='utf8') as f:
        for row in sorted(cases,key=lambda r:r['id']):f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    for fixed,r6,label in [(True,True,'Fixed_and_R6_correct'),(False,True,'R6_only_correct'),(True,False,'Fixed_only_correct'),(False,False,'Fixed_and_R6_wrong')]:
        group=[r for r in data if r['correct']['fixed']==fixed and r['correct']['M0_R6']==r6];c=sum(r['correct'][primary] for r in group)
        strata.append(dict(stratum=label,n=len(group),new_correct=c,new_wrong=len(group)-c,net_vs_R6=c-(len(group) if r6 else 0)))
    csvout('results/r11_rescue_retention.csv',strata)
    costs=[]
    for account in ['r7','r9','r10','r11']:
        ledger=read('logs/'+account+'_gpu_budget.json')
        for run in ledger['runs']:costs.append(dict(account=account,stage=run['name'],status=run['status'],GPU_worker_seconds=run['gpu_worker_seconds'],GPU_worker_hours=run['gpu_worker_seconds']/3600,
            note='Same original21600s cap; includes loading, failed/cancelled attempts and export. Collection cache reuse is not deployment latency.'))
    total=sum(r['GPU_worker_seconds'] for r in costs);assert total<=21600
    for p in ['logs/slcd_gpu_budget.json','logs/gpu_budget.json']:
        v=read(p);costs.append(dict(account='historical_not_added_to_new_cap',stage=p,status='preserved',GPU_worker_seconds=v['used_seconds'],GPU_worker_hours=v['used_seconds']/3600,note='Historical ledgers can overlap and include other work; do not naively add to each other or treat as isolated cache-only cost.'))
    csvout('results/r11_compute_costs.csv',costs)
    norm=comp[primary+'_prior_norm'];static=comp['robust_cyclic_native_axis' if primary=='robust_cyclic_loop_axis' else 'single_native_axis'];joint=comp['cyclic_joint4_only' if primary=='robust_cyclic_loop_axis' else 'joint4_only']
    direction_supported=norm['net']>0 and norm['exact_mcnemar_p']<.05 and norm['paired_bootstrap_95_lower_pp']>0
    loop_supported=static['net']>0 and static['exact_mcnemar_p']<.05 and static['paired_bootstrap_95_lower_pp']>0
    direction_note='方向修正在该历史基准上的主要匹配幅度对照比较得到支持，仍不等于语义轴已被识别。' if direction_supported else '未证明方向修正具有独立于幅度变化的收益；不能把点估计或随机对照差异当作机制成立。'
    loop_note='循环差轴相对静态Native轴的该配对比较支持其特定作用，但不构成新原理或一般化证明。' if loop_supported else '循环差轴未显示相对静态Native轴的可靠优势，循环特有解释仍缺乏证据。'
    practical_note=(f"对应的联合视图单独评分为{joint['reference_correct']}/1172，高于R11的{m['correct']}。本轮不能推荐R11作为整体准确率方案；单独联合视图的延迟未单测，不声称其具有已测速度优势。" if joint['net']<0 else '联合视图单独评分同列；本轮没有证实跨任务实用优势。')
    displayed=[dict(reference=r['reference'],correct=r['reference_correct'],R11_net=r['net'],rescue=r['rescue'],damage=r['damage'],p=f"{r['exact_mcnemar_p']:.6g}",paired95CI_pp=f"[{r['paired_bootstrap_95_lower_pp']:.4f},{r['paired_bootstrap_95_upper_pp']:.4f}]") for r in arc['comparisons']]
    report=f'''# R11：完整ARC-Challenge实验报告

冻结主方法 **{primary}**：**{m['correct']}/1172（{100*m['accuracy']:.4f}%）**；R6为736/1172。救回{m['rescue']}、误伤{m['damage']}，净变化{m['net']:+d}（{m['delta_pp']:+.4f}pp）。exact McNemar p={m['exact_mcnemar_p']:.6g}，20,000次题级配对bootstrap95%CI [{m['paired_bootstrap_95_lower_pp']:.4f},{m['paired_bootstrap_95_upper_pp']:.4f}]pp，seed2021。达到751：{m['correct']>=751}。

{direction_note}{loop_note} {practical_note}

## 协议、冻结与选择

原始Ouro-1.4B revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1，BF16/GB10，无训练、反向传播、权重更新、adapter或learned gate。原R6主提示、固定25示例、完整答案续写和字符分母保持。辅助视图保留同25个示例身份、顺序、题干和正确答案，但添加完整选项并改为字母答题格式；这属于辅助提示变化。候选按全词表log-softmax的完整目标LL计分，没有把Fixed替换为候选LL线性外推。

R7完成原token词表前缀投影实验，冻结M2为740/1172，未得到显著增益。用户要求继续方法研究、停止外部任务后，R8八种几何保护和R9逆向视图均未过门槛；R10直接联合融合被更强的joint-only诊断支配，明确在开发反馈后增加归因停止规则，未称原数值门槛失败。R11转而保护先验与当前联合循环变化轴的重合部分，不直接加入joint-only分类分数。

R11先在任何效果反馈前冻结两主候选及六控制；数值审查前后协议/源码版本均保留。576开发之后冻结完整实现，再用517确认及299验证。合并训练R6=705，单参考714，稳健716；验证单参考176、稳健178、R6=174、稳健幅度控制179。按公开排序规则选稳健方法，非晋级控制未被换成测试赢家。新完整测试前公开最终锁commit6a7ccf081981377ad2e8f1929d2ecccfd18651d2；最终锁SHA256为`{sha('configs/r11_FINAL_METHOD_LOCK.json')}`。

## 完整测试：预设主对比为R11对R6

{table(displayed,['reference','correct','R11_net','rescue','damage','p','paired95CI_pp'])}

该表包含主对比和非晋级诊断，未按测试结果择优。历史ARC与开发数据均已暴露；这些条件配对统计未校正跨轮适应性研究，不是新盲测或独立交叉验证。

![Results](r11_arc_results.png)

{table(strata,['stratum','n','new_correct','new_wrong','net_vs_R6'])}

R6相对Fixed的35道独有纠错保留{arc['retention']['retained_R6_special']}道。训练39道、开发21道是不同集合，分别报告。逐题分数/转移及所有预测变化的失败案例完整保存，例子不按margin好看程度挑选。

![Paired](r11_paired_comparisons.png)

## 方法与机制的范围

A=F+Q4−Q5，e=Fprior−P4，E=center(e)，g=center(F−Q4)。当前问题的联合选项前缀给出映回原索引的S_j=center(J4_j−J1_j)。保护p=b*mean(S_j)，b=<E,mean(S_j)>/mean(||S_j||²)；剩余v=center(e−p)，alpha=clip(<g,v>/<v,v>,0,1)，输出A−alpha*(e−p)。系数来自当前无标签前向，未用gold或数据集拟合控制器。

单轴是正交保护，稳健版本按参考变异削弱保护，不保证剩余向量正交，也不保证保留R6 margin。零轴精确回到冻结R6，零剩余先验回到A1。计算仍在候选分数空间；循环差是否代表有用语义是待检验假设。对全局S取反，保护子空间本身不区分其质量方向，不能宣称自动识别了改善与损害。

开发集后验审计复用全部真实循环上下文：1728次相关循环扰动中，单参考39次变化、涉及28题；稳健0次，映回分数最大差1.55e-15。仅覆盖循环移位，不能宣称一般排列不变性。参考平均和稳健分母的作用没有在R11单独完全隔离。开发7道稳健救回中，6道保护分量改善固定旧R6对手margin，3道系数分量改善，2道校正范数增加；代数分解有交互/顺序依赖，不是唯一的语义或隐藏状态因果解释。后验诊断未改变冻结方法。

![Repair](r11_repair_outcomes.png)

## 正确性、成本与结论

CPU数学/退化情况检查通过；真实两题辅助K4对历史K5相应1/4/Fixed读出以及新CPU评分全零误差。旧权重/词表/示例和原R6数字来源逐SHA绑定；完整test逐候选评分、预测、正确性及配对统计由独立验收重建。旧target哈希只覆盖原始顺序n个候选；全部移位额外核验request哈希、计数和LL，不伪称历史存了n² target哈希。

实际12题独立部署使用mainK5/N0priorK4/auxK4，没有KV、诊断hook或历史状态，完整分数均精确复建。后11题均值R11={profile['warm_mean_question_seconds']:.4f}s、R6={profile['R6_warm_mean_question_seconds']:.4f}s，比值{profile['time_ratio']:.4f}；顺序小样本测量不外推硬件通用速度。PyTorch峰值分配{profile['peak_cuda_allocated_bytes']/2**30:.4f}GiB，GB10为统一内存。完整benchmark采集复用原R7主/先验LL、重新前向辅助流，不能把采集耗时当独立部署耗时。

同一新增6小时账本合计GPU-worker {total:.3f}s（{total/3600:.4f}h），包含所有加载、失败/取消和导出。R7/R9/R10/R11账未重置；更早历史预算单列且不简单相加。完整代码、模型/数据/源码哈希、原始数值、图表、失败案例和重放验收记录保留。另两个数据集按用户要求停止。

本轮是否值得继续，应按对R6、幅度控制和静态轴的实际差异及额外计算判断。未得到支持的机制不作为已完成发现，也不因达到一个正确数门槛而声称新原理或实用优势。
'''
    (ROOT/'results/r11_REPORT_ZH.md').write_text(report,encoding='utf8')
    (ROOT/'r11_README.md').write_text(f'''# Frozen R11 ARC study

Primary {primary}: {m['correct']}/1172; R6=736, net{m['net']:+d}, rescue{m['rescue']}/damage{m['damage']}. Read [Chinese report](results/r11_REPORT_ZH.md), [locked primary](configs/r11_FINAL_METHOD_LOCK.json), [paired results](results/r11_arc_paired_statistics.csv), [costs](results/r11_compute_costs.csv).

Restore the inherited R6/R7 and development evidence named by artifacts/r11_INHERITED_EVIDENCE.json, then verify/restore the new multipart archive with python -B scripts/r11_restore.py --verify-only (omit --verify-only to create absent records; existing different files are refused). Exact numerical replay: python -B scripts/r11_replay.py --phase test. The pinned experiment environment is used for exact-score acceptance. No model loading is required for numerical replay; original-server live model-file checks use python -B scripts/r11_accept.py. Public-records-only verification is explicitly weaker and writes a separate receipt.

Original model ID ByteDance/Ouro-1.4B, revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1; weights/env/SSH and raw hidden states are excluded. The small custom snapshot_metadata.json is released as artifacts/r11_MODEL_SNAPSHOT_METADATA.json; scripts/r11_prepare_model.py --prepare verifies every pinned model/tokenizer/source byte before creating missing metadata/runtime receipts. It never downloads or edits weights and never writes through external model symlinks. Omit --prepare for read-only verification. Inherited ARC text retains data/DATA_SOURCE_NOTICE.txt attribution. Auxiliary joint-option prompts are a disclosed change. Data histories were exposed; no blind-test claim. {direction_note}{loop_note} {practical_note}
''',encoding='utf8')
    (ROOT/'r11_SELF_ITERATION_LOG.md').write_text(f'''# R11 finite iteration

R7 prefix-token methods and controls were completed and frozen; old report740/1172 unchanged. R8/R9 failed bounded gates. R10 completed576 joint-view collection, then stopped heldout work after a disclosed additional attribution decision. R11 two main definitions/eight total configurations were published before effects, with one pre-outcome numerical guard clarification preserved as v0. Development single366/robust367 vsR6360; freeze fullimplementation before517 confirmation, exactfreshK4 parity2. Fulltrain single714/robust716 vs705; fixed two-candidate validation176/178 vs174 selected robust. Controls remained nonselectable, including norm179. Public final freeze preceded one new fullARC. Test yielded{m['correct']}/1172, no alternate winner/retuning. Profile12 and all attempts count under same6h cap. Development-only algebra/cyclic audits were posthoc and never changed the candidate. External evaluations stayed stopped. A status probe at935/1172 inadvertently displayed an existing partial aggregate stdout tail; this was documented and caused no method, coefficient, candidate or selection change.
''',encoding='utf8')
    # Standalone scientific plots use accepted numeric data only.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    counts={r['reference']:r['reference_correct'] for r in arc['comparisons']};counts[primary]=m['correct']
    order=['native4','fixed','adaptive','A1','M0_R6',primary,primary+'_prior_norm',primary+'_shuffle_update','robust_cyclic_native_axis' if primary=='robust_cyclic_loop_axis' else 'single_native_axis','cyclic_joint4_only' if primary=='robust_cyclic_loop_axis' else 'joint4_only']
    labels=['Native K4','Fixed','Adaptive','A1','R6','R11 primary','Matched correction norm','Shuffled correction','Static joint axis','Joint K4 alone (auxiliary prompt)']
    fig,ax=plt.subplots(figsize=(8.4,5));y=np.arange(len(order));ax.scatter([100*counts[n]/1172 for n in order],y,c=['#1b6e75' if n==primary else '#8b9eae' for n in order],s=45,zorder=3);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Accuracy (%), complete1172');ax.axvline(100*736/1172,c='#91493e',ls='--',lw=1);ax.set_title(f'Frozen R11: {m["correct"]}/1172; paired net vs R6 {m["net"]:+d}');fig.tight_layout()
    for i,n in enumerate(order):ax.annotate(f'{counts[n]}/1172 ({100*counts[n]/1172:.2f}%)',(100*counts[n]/1172,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=8)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'results/r11_arc_results.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    refs=['fixed','A1','M0_R6',primary+'_prior_norm','robust_cyclic_native_axis' if primary=='robust_cyclic_loop_axis' else 'single_native_axis'];names=['Fixed','A1','R6','Matched correction norm','Static joint axis'];values=[comp[k] for k in refs]
    fig,ax=plt.subplots(figsize=(8.2,4));ys=np.arange(len(refs));means=np.array([r['delta_pp'] for r in values]);low=np.array([r['paired_bootstrap_95_lower_pp'] for r in values]);high=np.array([r['paired_bootstrap_95_upper_pp'] for r in values]);ax.errorbar(means,ys,xerr=np.vstack([means-low,high-means]),fmt='o',color='#1b6e75',capsize=3);ax.set_yticks(ys,names);ax.invert_yaxis();ax.axvline(0,c='#666666',lw=1);ax.set_xlabel('R11 minus reference accuracy (pp); paired20k-bootstrap95% CI');fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'results/r11_paired_comparisons.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.7));axes[0].bar(['Rescued','Damaged'],[m['rescue'],m['damage']],color=['#1b6e75','#ac584e']);axes[0].set_ylabel('Questions vs R6');ret=arc['retention']['retained_R6_special'];axes[1].barh(['R6-only correct'],[ret],color='#1b6e75',label=f'Retained {ret}');axes[1].barh(['R6-only correct'],[35-ret],left=ret,color='#ac584e',label=f'Lost {35-ret}');axes[1].set_xlim(0,35);axes[1].set_xlabel('35 original R6 rescues vs Fixed');axes[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.18));fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'results/r11_repair_outcomes.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    save('results/r11_DELIVERABLES.json',dict(status='rendered_from_complete_accepted_ARC',primary=primary,n=1172,correct=m['correct'],direction_control_supported=direction_supported,
        loop_axis_vs_static_supported=loop_supported,semantic_axis_identified=False,all_new_GPU_worker_seconds=total,full_report_sha256=sha('results/r11_REPORT_ZH.md'),failure_cases=len(cases),
        external_tasks_stopped=True,goal_success_declared=False))
    print('R11_REPORT_FIGURES_CASES_RENDERED',m['correct'],len(cases),flush=True)


if __name__=='__main__':main()
