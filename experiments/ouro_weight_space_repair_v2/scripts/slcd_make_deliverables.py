"""Generate final reports and immutable-source archives; never changes methods."""
import csv,hashlib,json,math,time,zipfile
from pathlib import Path
from slcd_core import ROOT,read,rows,sha,save,writecsv

def table(records,columns):
    header='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    return header+'\n'.join('| '+' | '.join(str(r.get(c,'')) for c in columns)+' |' for r in records)+'\n'

def text(path,value):
    p=ROOT/path;assert p.parent.is_dir() and p.name.startswith('slcd_');p.write_text(value,encoding='utf8')

def logical_heads(mid,registry,visiting=()):
    """Mathematical formula dependencies, not measured execution or latency."""
    assert mid not in visiting,'Cyclic method definition'
    spec=registry[mid];kind=spec['kind'];path=visiting+(mid,)
    if kind=='baseline':
        method=spec['method']
        return {int(method.removeprefix('native_k'))} if method.startswith('native_k') else {1,4}
    if kind=='score_late':return {1,4}|({spec.get('late',5)} if spec['gamma'] else set())
    if kind=='trajectory_reference':return {4,spec['reference']}
    if kind=='token':
        r=spec['token_recipe'];heads={r.get('reference',1),r.get('final',4)}
        if r['kind']=='two_sided_logit' and r.get('gamma',0)>0:heads.add(r.get('late',5))
        return heads
    if kind=='mix':return logical_heads(spec['a'],registry,path)|logical_heads(spec['b'],registry,path)
    if kind=='gate':
        signals={'k4_k5_disagree':{4,5},'k4_cd_disagree':{1,4},'cd_agrees_k5':{1,4,5},
                 'k4_margin_below':{4},'cd_margin_below':{1,4},'late_opposes_cd':{1,4,5}}
        return logical_heads(spec['a'],registry,path)|logical_heads(spec['fallback'],registry,path)|signals[spec['signal']]
    raise ValueError('Unrecognized reporting dependency: '+kind)

def main():
    arc=read('results/slcd_ARC_TEST_COMPLETE.json');assert arc['status']=='complete' and arc['n']==1172
    lock=read('FINAL_METHOD_LOCK.json');assert arc['final_lock_sha256']==sha('FINAL_METHOD_LOCK.json')
    passed=arc['passed_absolute_gate'];primary=arc['primary'];final={r['method']:r for r in arc['results']};p=final[primary]
    assert primary==lock['primary'] and passed==arc['other_benchmarks_authorized']==(p['correct']>717)
    external=[];external_incomplete=[]
    if passed:
        for task,n in [('hellaswag',10042),('openbookqa',500)]:
            if (ROOT/f'results/slcd_{task}_COMPLETE.json').exists():
                item=read(f'results/slcd_{task}_COMPLETE.json')
                assert item['status']=='complete' and item['n']==n
                assert sha(item['score_file'])==item['score_file_sha256']
                protocol=read(f'configs/slcd_{task}_PROTOCOL.json')
                assert sha(f'configs/slcd_{task}_PROTOCOL.json')==item['protocol_sha256']
                assert protocol['final_method_lock_sha256']==sha('FINAL_METHOD_LOCK.json')
                external.append(item)
            else:
                reason=read(f'results/slcd_{task}_NOT_COMPLETE.json')
                external_incomplete.append(dict(task=task,status=reason['status'],reason=reason['reason']))
    else:
        assert not (ROOT/'results/slcd_hellaswag_COMPLETE.json').exists() and not (ROOT/'results/slcd_openbookqa_COMPLETE.json').exists()
    budget=read('logs/slcd_gpu_budget.json');assert budget['used_seconds']<=21600 and all(r['status'] in ['complete','failed'] for r in budget['runs'])
    assert abs(sum(r['gpu_worker_seconds'] for r in budget['runs'])-budget['used_seconds'])<1e-6
    costs=[dict(job=r['name'],status=r['status'],actual_gpu_worker_seconds=r['gpu_worker_seconds'],gpu_worker_hours=r['gpu_worker_seconds']/3600,exit_code=r.get('exit_code'),limit_seconds=r['limit_seconds']) for r in budget['runs']]
    writecsv('results/slcd_compute_costs.csv',costs)
    all_dev=[]
    for name in ['slcd_iteration1.json','slcd_iteration2_token.json','slcd_candidate_pilot.json','slcd_iteration3.json','slcd_iteration4.json','slcd_iteration5.json']:
        all_dev+=read('results/'+name)['results']
    initial_native={s:next(r for r in all_dev if r['method']=='native4' and r['split']==s) for s in ['discovery','selection']}
    hybrid_confirmation=[r for r in all_dev if r['split']=='confirmation' and r['method'].startswith('H_')]
    hybrid_summary=('确认训练集中，已评估的'+str(len(hybrid_confirmation))+'个混合规则分别有'+
        str(sum(r['net_correct']>0 for r in hybrid_confirmation))+'个正净收益、'+
        str(sum(r['net_correct']==0 for r in hybrid_confirmation))+'个持平、'+
        str(sum(r['net_correct']<0 for r in hybrid_confirmation))+'个负净收益。')
    confirmation_notes='；'.join(f"{r['method']}净{r['net_correct']:+d}题" for r in all_dev
        if r['split']=='confirmation' and r['method'] in ['A1_g1p0','Ref2_w0p5'])
    # Preserve source-stage provenance; repeated control rows remain explicit.
    writecsv('results/slcd_development_results.csv',all_dev)
    ablations=[r for r in all_dev if r['split'] in ['discovery','selection']]
    writecsv('results/slcd_ablation_results.csv',ablations)
    validation=read('results/slcd_validation_results.json');vl=read('configs/slcd_VALIDATION_LOCK.json')
    registry=read('configs/slcd_METHODS.json')
    dev=[]
    for mid in ['fixed']+read('configs/slcd_CONFIRMATION_METHOD_LOCK.json')['methods']:
        out=dict(method=mid)
        for split in ['discovery','selection','confirmation']:
            found=[r for r in all_dev if r['method']==mid and r['split']==split]
            if found:
                r=found[-1];out[split]=f"{r['correct']}/{r['n']} ({r['net_correct']:+d})"
        dev.append(out)
    fp=lambda s:hashlib.sha256(' '.join(s.casefold().split()).encode()).hexdigest()
    sets={s:{fp(r['question']) for r in rows('data/slcd_'+s+'.jsonl')} for s in ['discovery','selection','confirmation','validation','test']}
    demos={fp(r['question']) for r in read('data/slcd_fixed_demonstrations.json')['documents']}
    overlap={a+'__'+b:len(sets[a]&sets[b]) for a in sets for b in sets if a<b}
    overlap.update({'demos__'+s:len(demos&sets[s]) for s in sets})
    save('results/slcd_SPLIT_OVERLAP_AUDIT.json',dict(status='complete_after_final_lock',fingerprint='casefold then collapse whitespace; question only',counts=overlap,no_posthoc_test_filter=True))
    # Compare all previous task artifacts against the untouched server baseline.
    original=read('logs/slcd_PRESERVATION_BASELINE.json');mismatch=[]
    for rel,expected in original.items():
        f=ROOT/rel
        if not f.exists() or f.stat().st_size!=expected['bytes'] or f.stat().st_mtime_ns!=expected['mtime_ns']:mismatch.append(rel)
    model_sha=sha('models/Ouro-1.4B/model.safetensors')
    assert not mismatch and model_sha==lock['model_safetensors_sha256'],mismatch
    save('results/slcd_PRESERVATION_AUDIT.json',dict(status='passed',previous_files_checked=len(original),changed_files=mismatch,model_safetensors_sha256=model_sha,weights_modified=False,scope='prior file sizes and nanosecond mtimes; full model byte SHA; excludes caches/runtime temp'))
    if (ROOT/'results/slcd_tokens_test_final_COMPLETE.json').exists():test_replay='1172题新鲜轨迹及8种原始基线全量精确重放'
    else:test_replay='复用兼容的1172题原始完整候选分数，CPU执行冻结公式；本轮test GPU重放覆盖为0'
    external_lines=[]
    for e in external:
        es={r['method']:r for r in e['results']};v=es[primary];b=es['fixed']
        external_lines.append(dict(task=e['task'],n=e['n'],Fixed=b['correct'],selected=v['correct'],net=v['net_correct'],delta_pp=round(v['delta_accuracy_pp'],4),p=round(v['exact_mcnemar_p'],6),CI=f"[{v['paired_bootstrap_95ci_low_pp']:.3f}, {v['paired_bootstrap_95ci_high_pp']:.3f}]"))
    q8=table(external_lines,['task','n','Fixed','selected','net','delta_pp','p','CI']) if external_lines else (
        'ARC主方法通过执行门槛，但没有外部任务的完整结果；不能据此判断跨任务效果。\n' if passed else
        '主方法未超过717题，按预定停止规则未运行HellaSwag、OpenBookQA或其他benchmark。\n')
    if external_incomplete:
        q8+='\n外部任务未完成/未运行状态（不报告部分样本准确率）：\n'+table(external_incomplete,['task','status','reason'])
    statistically_supported=p['paired_bootstrap_95ci_low_pp']>0 and p['exact_mcnemar_p']<.05
    outcome='超过717的执行门槛' if passed else '未超过717的执行门槛'
    finding='有正向配对统计支持' if statistically_supported else '没有获得可靠的配对统计支持'
    selected=lock['method_registry'][primary]
    logical_costs=[]
    for result in arc['results']:
        mid=result['method'];heads=logical_heads(mid,lock['method_registry'])
        depth=lock['method_registry'][mid]['required_depth'];assert max(heads)==depth
        logical_costs.append(dict(method=mid,logical_depth=depth,additional_loops_vs_fixed=depth-4,
            loop_work_ratio_vs_fixed=depth/4,loop_work_increase_percent=100*(depth/4-1),
            mathematical_head_readouts=len(heads),head_depths=','.join('K'+str(k) for k in sorted(heads)),
            additional_head_readouts_vs_fixed=len(heads)-2,actual_shared_depth=6,standalone_latency_measured=False))
    writecsv('results/slcd_logical_costs.csv',logical_costs)
    selected_cost=next(c for c in logical_costs if c['method']==primary)
    cost_explanation=(f"主方法公式需要{selected_cost['head_depths']}共{selected_cost['mathematical_head_readouts']}次head读出，Fixed为K1/K4共2次。"
        f"相对Fixed增加{selected_cost['additional_loops_vs_fixed']:+d}个循环，数学循环工作量比为{selected_cost['loop_work_ratio_vs_fixed']:.2f}，"
        f"即{selected_cost['loop_work_increase_percent']:+.0f}%。该比值只统计相同输入下共享循环块的调用次数，既不是整体FLOPs比，也不是实测延迟或吞吐比。")
    exposure_note=('本轮最终选中了历史已曝光726/1172的同一A1 gamma1配置。这是知情背景下的再选择/回归核对，不是独立发现或无污染验证。'
                   if primary=='A1_g1p0' else '历史A1 gamma1的726/1172已曝光，形式化训练/验证选择不能消除这类先验污染。')
    report=f'''# Training-free LoopCD 自主迭代实验报告

本轮在原V2文件夹内完成，所有新文件使用slcd_前缀；不覆盖原V2报告、结果或权重。模型为原始Ouro-1.4B，未训练、未编辑权重。最终主方法 `{primary}`：**{p['correct']}/1172（{100*p['accuracy']:.3f}%）**，Fixed为717/1172，净{p['net_correct']:+d}题（{p['delta_accuracy_pp']:+.3f}pp），{outcome}，{finding}。

## Q1. 原始LoopCD为什么产生有害改答？

Fixed把早期到K4的logit差继续外推，但该方向并非每个候选或样本都与正确答案一致。训练discovery384中，相对K4有{initial_native['discovery']['regressions']}次纠错、{initial_native['discovery']['corrections']}次退化；selection192中为{initial_native['selection']['regressions']}/{initial_native['selection']['corrections']}。候选分差及跨循环一致性是本轮检验的退化识别信号，其稳定性由确认训练集结果衡量。这个实验支持“统一外推会改变部分已正确排序”的操作性解释，不能据此断言某种隐藏状态因果机制。对gold margin的改善也不等同于top1正确数改善。

## Q2. 实现了哪些方法？

共注册{len(registry)}种配置（含基线与对照），研究5轮：A1字符归一化完整候选分数的晚期修正；A2全词表token logits修正后重新log-softmax；相对最终logits的正交/平行分解，centered和raw对照；确定性候选门控；基于失败模式的最小混合；gamma局部邻域、guidance强度和参考循环改变。没有训练门控器、拟合分类器或改动模型权重。A1与A2不等价。全部实验参数在configs/slcd_METHODS.json中。

正交投影采用a=z4、d=z4-z1、p=a(d·a)/(a·a)、o=d-p；centered版本先减各自词表均值，以获得softmax平移不变性；raw版本不中心化。原论文§5.2/F.2没有明确规定中心化，不能把这里的centered约定称为作者精确实现。[原论文](https://arxiv.org/html/2610.02185v1#S5.SS2)。零范数时p=0，无epsilon ridge。

## Q3. 自主迭代发现了什么？

初始A1和A2小系数的训练收益促使继续分析退化。训练特征提示K4/K5一致性及Fixed候选分差可能保护原答案，于是引入单一门控。{hybrid_summary}确认训练集简单方法结果：{confirmation_notes}。正交/平行引导的两训练子集结果完整保留在消融表，未把单一子集的上升当作稳健优势。本轮形式化选择使用登记的训练/validation指标，最终锁定后不修改参数；这不证明研究者未受已知历史aggregate影响。{exposure_note}

{table(dev,['method','discovery','selection','confirmation'])}

括号为相对Fixed的净纠错。517题在反馈后参与晋级，属于选择数据。前3晋级规则在结果揭示前写入slcd_ADVANCEMENT_POLICY.json。

## Q4. 开发数据上最佳方法与最终选择是什么？

旧discovery上最高的混合规则不等于最可信方法。确认训练集按预定排序晋级 `{', '.join(vl['finalists'])}`；随后只进行一次官方validation299比较，按正确数、三训练子集最差比例净收益、较少的纠错与退化总数、逻辑深度、复杂度和ID顺序决定主方法。Fixed/Adaptive只作比较，不在新方法主候选中；没有secondary。

{table([dict(method=r['method'],correct=r['correct'],n=r['n'],net=r['net_correct'],R=r['corrections'],D=r['regressions']) for r in validation['results']],['method','correct','n','net','R','D'])}

最终算法及所有依赖：
```json
{json.dumps(selected,ensure_ascii=False,indent=2)}
```
完整递归定义、数值系数、source/model/tokenizer/data/prompt hashes及测试授权见根目录FINAL_METHOD_LOCK.json。该锁在访问本轮测试分数前创建。

## Q5. 完整ARC是否超过717题？

{table([dict(method=r['method'],correct=r['correct'],n=r['n'],accuracy_percent=f"{100*r['accuracy']:.3f}",net=r['net_correct']) for r in arc['results']],['method','correct','n','accuracy_percent','net'])}

全部1172题，固定同一组25-shot，seed2021，原始BF16 checkpoint，lm-eval0.4.9完整候选条件似然，按答案字符数归一化、首个argmax打破平局。{test_replay}。A1的线性修正是排名分数，不能解释为校准的token对数概率。模型revision：{lock['model_revision']}。

## Q6. 纠错和新增错误各有多少？

主方法相对Fixed：纠错{p['corrections']}题，退化{p['regressions']}题，净{p['net_correct']:+d}题；全部改答{p['answer_changes']}题。错误到错误也计入改答，不能混入纠错计数。逐题完整转移及分差见slcd_paired_transitions.csv；全部原始候选分数与提示在记录归档。

## Q7. 统计和计算上是否值得？

主比较为唯一冻结主方法vsFixed。配对50,000次题级bootstrap的95%差值区间为[{p['paired_bootstrap_95ci_low_pp']:.3f}, {p['paired_bootstrap_95ci_high_pp']:.3f}]pp；双侧exact McNemar p={p['exact_mcnemar_p']:.6g}。超过717只是后续benchmark执行条件，并不等同于显著优越。训练/validation经过选择，不把其未校正p值作为确认性证据；次要比较未作多重显著性声明。

逐方法数学成本如下，完整字段见slcd_logical_costs.csv：

{table(logical_costs,['method','logical_depth','additional_loops_vs_fixed','loop_work_ratio_vs_fixed','mathematical_head_readouts','head_depths','actual_shared_depth'])}

{cost_explanation}

本轮GPU-worker累计{budget['used_seconds']:.3f}秒（{budget['used_seconds']/3600:.4f}小时），包括所有实际worker和失败尝试，不混入旧V1/V2预算。预检失败但未启动GPU的命令不收费。主方法数学所需深度{lock['logical_loop_depth']}，而本实现统一实际执行K6，额外头读取也有成本。源码门控公共接口还会访问K5；不能把数学深度当已实现的K4部署。没有独立生产路径测速，禁止声称共享轨迹评估带来延迟改善。

## Q8. 是否跨benchmark泛化？

{q8}
其他任务只有完整ARC主方法通过门槛后才获授权；如运行，均使用lm-eval0.4.9官方任务定义、固定0-shot和acc_norm，四种方法共享同一prompt，算法及系数保持冻结。这与ARC25-shot不同，不能直接跨任务比较绝对准确率。新任务无独立历史参考，全量同轨迹公平比较不被包装成独立baseline复现。

## Q9. 是否继续？

{'存在正向点估计，但是否值得扩大需结合配对区间和外部任务结果；不能把一次小幅回归集提高解释为普遍算法改进。' if passed else '本轮没有找到超过原始Fixed的冻结方法，按预定规则停止。现有证据不足以支持继续堆叠门控或扩大同一小网格；应保留负结果并重新提出能在独立任务上检验的假设。'}

当前研究的可交付价值是完整分数复用、token/score修正区分、精确对照和被否定的迁移假设。是否存在可推广方法需要独立未用数据的后续验证；本轮不将新模型、Loop设计或大规模训练加入范围。

## 限制、完整性和复现

- ARC test以及官方validation在历史任务中已经使用，均不能称新的盲测。历史A1 gamma1的726已暴露，是先验污染；本轮方法选择仅使用训练/validation证据，但无法消除研究者已知aggregate的影响。
- 训练按问题文本去重并排除全部25例；三个训练子集相互不重叠。官方数据跨集合文本重复审计在最终冻结后运行，见slcd_SPLIT_OVERLAP_AUDIT.json，不事后删测试题。计数：`{json.dumps(overlap,sort_keys=True)}`。
- 之前{len(original)}个文件的size/mtime和完整模型字节SHA验证通过；未改动旧实验。保留旧报告文件名，新交付使用slcd_前缀以满足同一文件夹约束。
- 初始worker收据缺口由独立CPU source/state/target审计补齐；绑定时间明确标注为事后。后续v2 worker要求完整cache授权及held-out hash锁。confirmation没有旧独立参考，只有same-forward replay，不能声称独立parity。训练576与validation299的8种基线均精确相等；test重放覆盖见BASELINE_REUSE_AUDIT。
- slcd_records.zip包含全量新提示、逐候选分数、协议、数据hash、日志和可恢复记录；体积较大的原始hidden-state .pt保存在服务器同一results目录，索引包含每个文件SHA；它们是加速缓存，不是复现必需的权重。原始模型从固定revision取得，仓库不复制模型权重。
- 数据来源与许可沿用data/DATA_SOURCE_NOTICE.txt；额外任务来源commit、raw SHA及官方task source SHA在各PROTOCOL收据中。所有新图由实际表格生成。
'''
    text('results/slcd_REPORT_ZH.md',report)
    readme=f'''# Self-improving training-free LoopCD (original Ouro weights)

Final: **{primary}: {p['correct']}/1172 vs Fixed717**; net{p['net_correct']:+d}, paired95%CI[{p['paired_bootstrap_95ci_low_pp']:.3f},{p['paired_bootstrap_95ci_high_pp']:.3f}]pp, exactMcNemar p={p['exact_mcnemar_p']:.6g}.
Actual additional GPU-worker: {budget['used_seconds']/3600:.4f}h /6h. This is a previously used regression benchmark, not blind.

Same existing V2 folder; all old files preserved. Entry points:
- [Chinese report](results/slcd_REPORT_ZH.md)
- [Final method lock](FINAL_METHOD_LOCK.json)
- [Iteration log](SELF_ITERATION_LOG.md)
- [Full ARC table](results/slcd_arc_full_test_results.csv)
- [Paired transitions](results/slcd_paired_transitions.csv)
- [Development](results/slcd_development_results.csv), [ablations](results/slcd_ablation_results.csv), [worker costs](results/slcd_compute_costs.csv), [mathematical loop/head costs](results/slcd_logical_costs.csv)
- [Accuracy](results/slcd_accuracy_comparison.pdf), [rescue/regression](results/slcd_rescue_regression_analysis.pdf)

Model: ByteDance/Ouro-1.4B@{lock['model_revision']}; original BF16 checkpoint SHA{lock['model_safetensors_sha256']}. No training or edits.

Logical loop ratio versus Fixed: {selected_cost['loop_work_ratio_vs_fixed']:.2f}; head readouts: {selected_cost['mathematical_head_readouts']} versus2. Shared data collection executesK6; no standalone latency improvement is established. The selected A1 gamma1 configuration was historically exposed at726/1172 when applicable; reselecting it is not an independent or uncontaminated discovery.

Reproduction requires the pinned model and the original V1/V2 environment/utility code retained by this branch. Restore compressed records into this existing directory using `python -B scripts/slcd_restore_records.py` (SHA checked; refuses different existing files). Then inspect all locks before inference. Raw state caches remain on server; regenerate them from original model if absent. The original weighted-edit bundles are not inputs to this task.

Development order: iteration1 score rules → original-state token worker → token/orthogonal and failure-feature audit → hybrids → local token neighbors → frozen517 confirmation → frozen top3 validation → one final method lock → full ARC → conditional external tasks only if primary>717. All commands and recipes are in scripts/slcd_*, configs/slcd_* and SELF_ITERATION_LOG.md. Use scripts/slcd_supervise.py for every GPU worker to enforce the independent21600s ledger.

Final scoring entry: `env/bin/python scripts/slcd_finalize.py test`. Token methods first require the locked test worker `scripts/slcd_token_worker_v2.py --split test --job final --recipes configs/slcd_TOKEN_TEST.json`; score-only methods reuse compatible saved baselines. Running against completed immutable receipts validates/reuses outputs. Do not invoke selection again after the final lock or tune against test.

Frozen other-task command (only after ARC gate): `env/bin/python scripts/slcd_external_worker.py --task hellaswag` (and openbookqa if budget). Official harness0.4.9 task config, zero-shot, acc_norm, no retuning. Shared K6 collection gives no standalone runtime speed claim.

{'External completion or non-completion receipts are under results/slcd_hellaswag* and slcd_openbookqa*; only complete task results are reported.' if passed else 'ARC gate failed. No other benchmark was run.'}
'''
    text('slcd_README.md',readme)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':['DejaVu Sans'],'font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    labels=['Native K4','Fixed','Adaptive','Selected'];colors=['#CFCECE','#767676','#42949E','#0F4D92'];rs=arc['results']
    fig,ax=plt.subplots(figsize=(8,4.5));bars=ax.bar(labels,[100*r['accuracy'] for r in rs],color=colors,edgecolor='#272727',linewidth=1)
    for b,r in zip(bars,rs):ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{r['correct']}/1172\n{100*r['accuracy']:.3f}%",ha='center',fontsize=11)
    ax.set_ylim(0,75);ax.set_ylabel('ARC-Challenge accuracy (%)');ax.set_title('Frozen methods; same fixed 25-shot prompts');fig.tight_layout(pad=1.2)
    fig.savefig(ROOT/'results/slcd_accuracy_comparison.pdf');fig.savefig(ROOT/'results/slcd_accuracy_comparison.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.3));inds=np.arange(len(rs));axes[0].bar(inds-.18,[r['corrections'] for r in rs],width=.36,label='Rescues',color='#8BCF8B',edgecolor='#272727');axes[0].bar(inds+.18,[-r['regressions'] for r in rs],width=.36,label='Regressions',color='#E9A6A1',edgecolor='#272727');axes[0].axhline(0,color='#767676',lw=.8);axes[0].set_xticks(inds,labels,rotation=20);axes[0].set_ylabel('Questions vs Fixed');axes[0].legend(fontsize=10)
    vals=np.asarray([r['delta_accuracy_pp'] for r in rs]);lo=np.asarray([r['paired_bootstrap_95ci_low_pp'] for r in rs]);hi=np.asarray([r['paired_bootstrap_95ci_high_pp'] for r in rs]);axes[1].vlines(inds,lo,hi,color='#0F4D92');axes[1].hlines(lo,inds-.06,inds+.06,color='#0F4D92');axes[1].hlines(hi,inds-.06,inds+.06,color='#0F4D92');axes[1].scatter(inds,vals,color='#0F4D92');axes[1].axhline(0,color='#767676',lw=.8);axes[1].set_xticks(inds,labels,rotation=20);axes[1].set_ylabel('Accuracy difference (pp)');axes[1].set_title('Paired bootstrap 95% CI',fontsize=12);fig.tight_layout(pad=1.2)
    fig.savefig(ROOT/'results/slcd_rescue_regression_analysis.pdf');fig.savefig(ROOT/'results/slcd_rescue_regression_analysis.png',dpi=200);plt.close(fig)
    # All record files remain flat in existing folders. Large state caches stay server-side.
    files=[]
    for folder in ['data','configs','results','astra']:
        files += [p for p in (ROOT/folder).glob('slcd_*') if p.is_file() and p.suffix not in ['.pt','.tmp','.png','.pdf'] and p.name not in ['slcd_records.zip','slcd_RECORDS_CONTENTS.json','slcd_ARTIFACT_INDEX.json']]
    files += [p for p in (ROOT/'logs').glob('slcd_*') if p.is_file() and p.name not in ['slcd_final_chain.out','slcd_final_chain.err'] and (p.name.endswith('_SUPERVISOR.json') or p.name in ['slcd_gpu_budget.json','slcd_PRESERVATION_BASELINE.json'] or p.suffix in ['.out','.err'])]
    index=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p.relative_to(ROOT))) for p in sorted(set(files))]
    save('artifacts/slcd_RECORDS_CONTENTS.json',dict(file_count=len(index),files=index))
    with zipfile.ZipFile(ROOT/'artifacts/slcd_records.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):z.write(p,p.relative_to(ROOT).as_posix())
    artifact_paths=[p for folder in ['scripts','configs','results','astra','artifacts'] for p in (ROOT/folder).glob('slcd_*') if p.is_file() and p.suffix not in ['.pt','.jsonl','.parquet'] and p.name!='slcd_ARTIFACT_INDEX.json']+[ROOT/'slcd_README.md',ROOT/'SELF_ITERATION_LOG.md',ROOT/'FINAL_METHOD_LOCK.json',ROOT/'slcd_TASK_ORIGINAL.md']
    index=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p.relative_to(ROOT))) for p in sorted(set(artifact_paths))]
    save('artifacts/slcd_ARTIFACT_INDEX.json',dict(status='complete',primary=primary,correct=final[primary]['correct'],record_archive_sha256=sha('artifacts/slcd_records.zip'),file_count=len(index),files=index))
    print(json.dumps(dict(status='complete',primary=primary,correct=final[primary]['correct'],gpu_worker_hours=budget['used_seconds']/3600,archive_bytes=(ROOT/'artifacts/slcd_records.zip').stat().st_size,indexed_files=len(index))))

if __name__=='__main__':main()
