"""Render evidence from completed artifacts; never chooses a test winner."""
import csv,json,time
from pathlib import Path
import numpy as np
from r7_core import ROOT,read,save,sha,csvout


def csvrows(name):
    with (ROOT/name).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def markdown_table(rows,columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(str(r.get(c,'')) for c in columns)+' |' for r in rows])


def textout(name,content):
    p=ROOT/name;assert p.name.startswith('r7_') and p.parent.is_dir();p.write_text(content.rstrip()+'\n',encoding='utf8')


def main():
    from r7_evaluate import rows,paired
    arc=read('results/r7_ARC_COMPLETE.json');assert arc['status']=='complete'
    lock=read('configs/r7_FINAL_METHOD_LOCK.json');primary=lock['primary'];assert arc['primary']==primary
    test=rows('test');train=[r for s in ['discovery','selection','confirmation'] for r in rows(s)]
    val=rows('validation');dev=csvrows('results/r7_development_results.csv');full=csvrows('results/r7_arc_full_test_results.csv')
    contrasts=[]
    for name,ref in [('M2_N0_token','M0_R6'),('M3_multi_token','M2_N0_token'),
            ('M2_N0_token','C_R6alpha_token'),('M3_multi_token','C_mean_denominator'),
            ('M3_multi_token','C_matched_guidance'),('M3_multi_token','C_shuffled_direction'),('M3_multi_token','C_no_late')]:
        contrasts.append(dict(split='train1093',**paired(train,name,reference=ref)))
    csvout('results/r7_mechanism_contrasts.csv',contrasts)
    selectivity=[]
    for row in train:
        for name in ['M2_N0_token','M3_multi_token']:
            before=row['correct']['M0_R6'];after=row['correct'][name]
            category=('Correct' if before else 'Wrong')+'->'+('Correct' if after else 'Wrong')
            for candidate,diagnostics in enumerate(row['token_diagnostics']):
                if name not in diagnostics:continue
                d=diagnostics[name];n=len(d['alpha']);assert n>0
                selectivity.append(dict(id=row['id'],method=name,candidate=candidate,transition=category,tokens=n,
                    alpha_mean=float(np.mean(d['alpha'])),alpha_zero_fraction=float(np.mean(np.asarray(d['alpha'])==0)),
                    alpha_one_fraction=float(np.mean(np.asarray(d['alpha'])==1)),
                    reference_variance_mean=float(np.mean(d['robust_denominator'])),
                    target_logit_shift_sum=float(np.sum(d['target_logit_shift'])),
                    normalizer_change_sum=float(np.sum(d['log_normalizer_change'])),
                    target_logprob_change_sum=float(np.sum(d['target_logprob_change'])),
                    numerical_identity_max_abs_error=float(np.max(np.abs(np.asarray(d['target_logit_shift'])-np.asarray(d['log_normalizer_change'])-np.asarray(d['target_logprob_change']))))))
    csvout('results/r7_selectivity_diagnostics.csv',selectivity)
    ledger=read('logs/r7_gpu_budget.json');assert ledger['used_seconds']<=21600
    costs=[]
    for run in ledger['runs']:
        costs.append(dict(account='R7_new',stage=run['name'],status=run['status'],GPU_worker_seconds=run['gpu_worker_seconds'],
            GPU_worker_hours=run['gpu_worker_seconds']/3600,phase=run['phase'],includes_failed_attempts=True,
            note='Wall time of bounded GPU worker, including loading, head/projection, CPU export and failure overhead'))
    for name in ['logs/slcd_gpu_budget.json','logs/gpu_budget.json']:
        old=read(name);costs.append(dict(account='historical_not_charged_to_R7',stage=name,status='preserved',
            GPU_worker_seconds=old.get('used_seconds'),GPU_worker_hours=old.get('used_seconds',0)/3600,
            note='Existing complete ledger; includes other research work. Not an isolated cache-generation estimate; do not sum overlapping scopes.'))
        for run in old.get('runs',[]):
            if 'collect' in run.get('name','') or 'confirmation' in run.get('name',''):
                seconds=run.get('gpu_worker_seconds')
                costs.append(dict(account='historical_cache_related_run_not_new_charge',stage=run.get('name'),status=run.get('status'),
                    GPU_worker_seconds=seconds,GPU_worker_hours=seconds/3600 if seconds is not None else None,
                    note='Explicit cache-related historical run; original discovery collection cannot be isolated from the supplied ledgers.'))
    csvout('results/r7_compute_costs.csv',costs)
    # Preservation verification reads bytes/statistics only, never old test labels.
    original=read('logs/r7_PRESERVATION_BASELINE.json');entries=original.get('files',original.get('records',original));changed=[]
    if isinstance(entries,dict):entries=[dict(path=k,**(v if isinstance(v,dict) else {'sha256':v})) for k,v in entries.items()]
    for entry in entries:
        relative=entry.get('path',entry.get('file'));p=ROOT/relative
        if not p.is_file():changed.append(relative);continue
        st=p.stat()
        if 'sha256' in entry:
            if sha(relative)!=entry['sha256']:changed.append(relative)
        elif st.st_size!=entry['bytes'] or st.st_mtime_ns!=entry['mtime_ns']:changed.append(relative)
    assert entries and not changed,('Pre-R7 preservation failed',changed)
    inherited=read('artifacts/slcd_R6_RECORDS_CONTENTS.json')['files']
    for entry in inherited:assert sha(entry['path'])==entry['sha256'],('Inherited R6 record changed',entry['path'])
    assert sha('configs/slcd_R6_FINAL_LOCK.json')=='6de605b885b4fb9ff72b273ecc0c0484e9fe72fc5d51d793b22367db879b7da7'
    save('results/r7_PRESERVATION_AUDIT.json',dict(status='passed',stat_snapshot_files=len(entries),R6_hash_checked_files=len(inherited),changed=changed,
        snapshot_sha256=sha('logs/r7_PRESERVATION_BASELINE.json'),scope='Size/mtime_ns of every enumerated pre-R7 file plus complete immutable R6 archive record hashes; environment/model symlink/cache/transport and nested initial-discovery hidden states excluded from snapshot'))
    parity=read('results/r7_baseline_parity.json');parity.update(status='passed_CPU_and_GPU',
        gpu_split_receipts={s:dict(complete_sha256=sha('results/r7_'+s+('_development_COMPLETE.json' if s in ['discovery','selection','confirmation'] else '_final_COMPLETE.json')),
            **{k:v for k,v in read('results/r7_'+s+('_development_COMPLETE.json' if s in ['discovery','selection','confirmation'] else '_final_COMPLETE.json')).items() if k in ['n','baseline_parity_questions','baseline_parity_max_error','baseline_parity_status','historical_states_reused','actual_main_depth','historical_main_state_collection_depth']}) for s in ['discovery','selection','confirmation','validation','test']},
        pilot_sha256=sha('results/r7_PILOT_COMPLETE.json'),KV_cache_rejected=True,
        KV_reason='Real discovery pilot failed zero-tolerance BF16 parity: max token error0.3139867783; original uncached path retained',
        mathematical_checks_sha256=sha('results/r7_MATH_CHECKS.json'))
    save('results/r7_baseline_parity.json',parity)
    oracle=csvrows('results/r7_oracle_alpha_diagnostics.csv');lim={k:sum(r['limitation']==k for r in oracle) for k in set(r['limitation'] for r in oracle)}
    profile=read('results/r7_DEPLOYMENT_PROFILE.json')
    # All plots use complete source artifacts and export standalone PNG/PDF.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    methods=list(test[0]['predictions']);numbers=[sum(r['correct'][m] for r in test) for m in methods]
    fig,ax=plt.subplots(figsize=(8,4.1));bars=ax.bar(methods,numbers,color=['#aab8c5']*(len(methods)-1)+['#1b6e75'])
    for bar,n in zip(bars,numbers):ax.text(bar.get_x()+bar.get_width()/2,n+1,str(n),ha='center',va='bottom')
    ax.axhline(736,color='#7a3d35',ls='--',lw=1,label='R6 gate736');ax.axhline(751,color='#615a83',ls=':',lw=1,label='Target751')
    ax.set_ylabel('Correct /1172');ax.set_ylim(max(0,min(numbers)-20),max(756,max(numbers)+12));ax.tick_params(axis='x',rotation=17);ax.legend(frameon=False,loc='upper left');fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'results/r7_arc_results.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    aggregate=[r for r in dev if r['split']=='train1093'];names=[r['method'] for r in aggregate]
    fig,ax=plt.subplots(figsize=(8,5));ys=np.arange(len(names));net=[int(r['net']) for r in aggregate]
    ax.barh(ys,net,color=['#1b6e75' if x>=0 else '#aa544c' for x in net]);ax.set_yticks(ys,names);ax.axvline(0,c='#333333',lw=.8);ax.set_xlabel('Net rescued questions vsR6, train1093');ax.invert_yaxis();fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'results/r7_development_ablation.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8));axes[0].hist([float(r['alpha']) for r in oracle],bins=np.linspace(0,1,21),color='#1b6e75');axes[0].set(xlabel='R6 alpha',ylabel='Questions /576')
    cats=sorted(lim);axes[1].barh(cats,[lim[k] for k in cats],color='#758da5');axes[1].set_xlabel('Questions');fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'results/r7_direction_diagnostics.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    mainrow=arc['metrics'];gate=arc['correct']>736;achieve=arc['correct']>=751
    main_train=next(r for r in aggregate if r['method']==primary)
    contrast_table=[{'contrast':r['method']+' vs '+r['reference'],'rescue':r['rescue'],'damage':r['damage'],'net':r['net']} for r in contrasts]
    testtable=[dict(method=r['method'],correct=r['correct'],accuracy=f"{100*float(r['accuracy']):.4f}%",rescue=r['rescue'],damage=r['damage'],net=r['net']) for r in full]
    comparison=next(r for r in contrasts if r['method']=='M3_multi_token' and r['reference']=='C_matched_guidance')
    shuffled=next(r for r in contrasts if r['method']=='M3_multi_token' and r['reference']=='C_shuffled_direction')
    external_completed=gate and all((ROOT/f'results/r7_{task}_COMPLETE.json').exists() for task in ['hellaswag','openbookqa'])
    external_section=''
    if external_completed:
        externals=[]
        for task in ['hellaswag','openbookqa']:
            receipt=read(f'results/r7_{task}_COMPLETE.json');assert receipt['status']=='complete' and receipt['primary']==primary
            assert receipt['n']=={'hellaswag':10042,'openbookqa':500}[task]
            assert sha(receipt['score_file'])==receipt['score_file_sha256']
            externals.append(dict(task=task,**receipt['metrics']))
        csvout('results/r7_external_results.csv',externals)
        external_note='同一冻结方法完成HellaSwag10042及OpenBookQA500，结果见迁移表；未使用外部结果调整方法。'
        external_section='\n## 同一冻结公式的外部迁移\n\n'+markdown_table(externals,['task','n','correct','reference_correct','rescue','damage','net','exact_mcnemar_p'])+'\n\n两个任务均保持原0-shot官方协议和字符分母。N0为Answer:；N1/N2按ARC测试前声明的字段规则，将官方processed文档的query（HellaSwag）或question_stem（OpenBookQA）替换为N/A或Unknown，并用官方fewshot_context(0)渲染。所有原始prompt及参考渲染逐题核验。M1仅需候选完整LL，复用已绑定历史主LL和N0LL，并在首/中/末三题独立前向精确核验；M2/M3需要完整词表，因此重新前向。历史复用成本及独立核验覆盖分别记录，不宣称三题核验覆盖全量。\n'
    else:external_note='ARC门槛已打开；外部基准结果尚未交付，当前只完成ARC阶段。' if gate else '未超过736，外部基准按预设规则停止；未改公式、参考或另换候选重测。'
    significance='配对检验未提供显著增益证据（p≥0.05）；点估计不能据此解释为稳定提升。' if mainrow['exact_mcnemar_p']>=.05 else '配对检验显示该历史测试集上的差异，仍需新数据复核其可迁移性。'
    direction_note='M3在开发集正确数上同时超过两个控制；这只是有限开发比较，仍不能据此宣称发现新机制。' if comparison['net']>0 and shuffled['net']>0 else 'M3未同时超过匹配guidance范数与置乱控制；本轮不能证明定向先验修正具有独立于幅度变化的优势。'
    report=f'''# Ouro R7：完整ARC实验报告

冻结主方法 **{primary}**：**{arc['correct']}/1172（{100*mainrow['accuracy']:.4f}%）**，R6为736/1172；净变化 **{mainrow['net']:+d}题**，新增纠错{mainrow['rescue']}、误伤{mainrow['damage']}。超过736：{'是' if gate else '否'}；达到751：{'是' if achieve else '否'}。{external_note}

主要配对差值95% bootstrap区间为[{mainrow['paired_bootstrap_95_lower_pp']:.4f}, {mainrow['paired_bootstrap_95_upper_pp']:.4f}]pp（20,000次，seed2021）；exact McNemar p={mainrow['exact_mcnemar_p']:.6g}。{significance}这些是历史已暴露test上的条件比较，区间没有校正历轮研究者适应性选择。

## 协议与选择

在原V2文件夹使用r7_命名空间；用户后续同文件夹要求覆盖原附件的子目录要求。固定Git起点81d1c83b79461bf69882a0ffd4cf39b38dc3ba36。原始Ouro-1.4B revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1，BF16、GB10、seed2021、同一25-shot、lm-eval0.4.9 acc_norm、完整续写及原字符分母。

先运行discovery384+selection192，冻结8种定义及排序程序后单次使用confirmation517；按预定train排序只保留两个固定候选进行validation299。最终以validation正确数、全train正确数、对R6误伤、参考数和ID依次排序选出一个主方法。原始数据均有历史使用，不是新盲测或独立交叉验证。

候选锁SHA256：`{sha('configs/r7_CANDIDATE_LOCK.json')}`；最终锁SHA256：`{sha('configs/r7_FINAL_METHOD_LOCK.json')}`。所有推理源代码、模型/tokenizer、数据、固定示例、参考和数值常量均被绑定。每个配置任一候选的自身参考前缀对齐失败即整题回退R6；模型/数值错误停止。推理函数不接收gold。

## 完整test结果：主要参照R6

{markdown_table(testtable,['method','correct','accuracy','rescue','damage','net'])}

主方法保留R6相对Fixed原有纠错中的{mainrow['retained_R6_vs_Fixed_rescues']}/{mainrow['total_R6_vs_Fixed_rescues']}题。逐题记录、五组配对转移、候选总分和token诊断保存在原始JSONL及r7_paired_transitions.csv。没有测试后更换赢家。

![ARC](r7_arc_results.png)

## R6瓶颈与参考审计

576题中R6正确360；216道错题中沿A−aE、a∈[0,1]的事后oracle只能救回32题，184题在既有方向空间中无法排第一。这揭示旧方向的容量限制，不意味着token新方向必然有效；oracle不参与推理或方法选择。

N0为字面量Answer:；N1/N2保留原25例和分隔符，仅通过结构化question字段覆盖为N/A或Unknown。当前问题答案及解释不进入参考。每个流独立encode_pair，对齐已消费的续写文本；预测token仍从真实流target索引读取。R6候选复制/删除依赖及oracle逐题记录均公开。删除原赢家的自然变化不计作排名不稳定，使用共享候选限制排名进行区分。

![Direction](r7_direction_diagnostics.png)

## token粒度、多参考与控制

{markdown_table(contrast_table,['contrast','rescue','damage','net'])}

表中均为完整train1093上的配对比较，属于开发证据。M2对R6同时改变token粒度、系数估计和重新归一化，因此不能把全部差值归因于单一因素；桥接控制固定整题alpha后在token空间重新归一化，提供中间对照。M3对M2比较引入N1/N2的稳健参考作用。{direction_note}

若冻结赢家为M1，现有匹配幅度及置乱控制定义在M3的token空间，不能据此隔离M1的候选级方向机制。M1相对R6仅支持这套多参考候选公式的配对效果，不能把其提升归因于更细的token校正。

M1是候选分数空间投影；M2/M3先在完整词表加权内积下解析投影，再完整log_softmax，最后才累加并除一次字符长度。两类归一化不同，不能相互替代。r7_selectivity_diagnostics.csv将实际目标logit位移、log-partition变化和目标log概率变化分开，检查三者恒等式；不把已读标签用于系数。

![Development](r7_development_ablation.png)

## 正确性与计算成本

20项CPU数学检查通过；真实模型无干预、N0先验、独立循环深度及各阶段历史基线精确parity见r7_baseline_parity.json。旧KV路径真实BF16单token最大差约0.314，未通过且停用。所有新模型前向使用原无mask、无KV的调用。开发复用历史K6目标状态并逐文件SHA/manifest绑定；这不计作本轮主上下文K5执行时间。

新增GPU-worker累计{ledger['used_seconds']:.3f}秒（{ledger['used_seconds']/3600:.4f}小时），包含加载、失败尝试、参考、head和导出；硬上限21600秒，开发阶段预留5400秒。旧预算未修改；完整历史ledger与可识别缓存采集条目单列，不宣称历史缓存免费，也不把包含其他工作的总额误当缓存专属成本。

独立部署测量使用真实主上下文K5与参考K4，不使用历史K6状态、KV或诊断hooks/CPU诊断数组导出。12道固定discovery题与已完成开发分数逐候选精确一致；模型加载{profile['load_seconds']:.3f}秒，首题{profile['first_question_seconds']:.3f}秒，后11题均值{profile['warm_mean_question_seconds']:.3f}秒/题。PyTorch峰值分配{profile['peak_cuda_allocated_bytes']/2**30:.3f}GiB；GB10为统一内存，此值不是独立显存总占用。参考前缀缓存未获批准，缓存预热收益不适用。

同一12题独立R6执行的后11题均值为{profile['R6_warm_mean_question_seconds']:.3f}秒/题，新方法为其{profile['warm_time_ratio_vs_R6']:.3f}倍。两组按顺序测量，结果仅代表该小样本部署路径，不外推为硬件通用速度结论。

## 研究结论

本轮应按完整test的净纠错、保留旧纠错和配对不确定性判断收益。开发方向容量诊断支持“旧候选方向存在局限”，不能单凭它宣称新的隐藏状态失效机制已被验证。token与多参考的实际增量、控制结果和额外成本如上；没有训练、权重更新、控制器、adapter或循环结构改造。{external_note}

后续只有在新数据或预先限定的机制干预下重复观察到“方向优于匹配幅度控制”，才有依据将选择性校正提升为机制结论。本次有限方法比较不构成未见研究的原创性证明。
{external_section}
'''
    textout('results/r7_REPORT_ZH.md',report)
    textout('r7_README.md',f'''# Ouro R7 prefix-conditioned prior projection

Primary **{primary}**, complete ARC-Challenge **{arc['correct']}/1172**; R6 **736/1172**, net **{mainrow['net']:+d}**. Target751 {'reached' if achieve else 'not reached'}. {external_note}

All work remains in the original V2 directory; new files user7_ names. Read [Chinese report](results/r7_REPORT_ZH.md), [locked method](configs/r7_FINAL_METHOD_LOCK.json), [test metrics](results/r7_arc_full_test_results.csv), [paired transitions](results/r7_paired_transitions.csv), [compute accounting](results/r7_compute_costs.csv).

Replay completed numerical records onCPU: `python -B scripts/r7_evaluate.py --stage arc`. Rebuilding scores requires the original pinned model, fixed manifests and environment from the base repository. Inference must use the original no-KV/no-mask BF16 path; the historical KV optimization failed parity. `r7_dispatch.py` refuses duplicate failed attempts and never selects an alternate test winner. `r7_deploy.py` separately profiles exact5/4 deployment against development scores.

The branch also carries the unchanged inheritedR6 multipart release. Restore its inputs with `python -B scripts/slcd_r6_restore_records.py`, then verify/restore new numerical evidence with `python -B scripts/r7_restore.py`. `--verify-only` checks every indexed byte without overwriting. Historical raw states are not required by `scripts/r7_replay_fresh.py`, which reruns the frozen formula from the original model and manifests and compares every scored vector to archived results.

Inherited ARC question/rendered-demo data retain [CC BY-SA4.0 attribution](data/DATA_SOURCE_NOTICE.txt). Model weights, environment, credentials, transport and raw historical hidden-state tensors are excluded from the public evidence archive; historical states remain regenerable under their SHA authorities. Dataset histories were already exposed; no blind-test claim.
''')
    textout('r7_SELF_ITERATION_LOG.md',f'''# R7 self-iteration log

1. CPU replay of384+192 restored R6=360/576; oracle path diagnosis found32 coefficient-reachable and184 direction-limited remaining errors. Fixed-prompt choice perturbations were audited. No confirmation/test labels were read.
2. Twenty CPU math checks passed. A real BF16 discovery pilot reproduced original scores and independent K1/4/5 exactly. KV cache parity failed (max token0.3139867783), so it was rejected. Two-question integration runs checked original/N0 sums and token invariants; source/data binding and receipt coverage were tightened before full development. All attempts are charged and retained.
3. The eight prescribed configurations ran on384+192; no coefficient grid and no extra mixture were added. All definitions and selection code were frozen before517 confirmation. Top two main configurations advanced once to299 validation, and {primary} was frozen before1172 test.
4. Test produced{arc['correct']}/1172, net{mainrow['net']:+d} vsR6. {'External gate opened; same locked method is required.' if gate else 'External gate closed; no alternate test winner or retuning.'} Separate12-question5/4 deployment profile matched development scores exactly. Formula revision rounds after first complete dev feedback:0, within maximum3.

This is a finite negative-or-positive experiment, not a search guarantee. Full original/source hashes, immutable stage locks, raw scores, diagnostics, logs and cost accounting accompany the report.
''')
    save('results/r7_DELIVERABLES.json',dict(status='rendered_from_complete_ARC',primary=primary,arc_correct=arc['correct'],
        external_gate_open=gate,external_completed=external_completed,final_method_lock_sha256=sha('configs/r7_FINAL_METHOD_LOCK.json'),
        report_sha256=sha('results/r7_REPORT_ZH.md'),compute_budget_used_seconds=ledger['used_seconds']))
    print(json.dumps(dict(status='report_rendered',primary=primary,arc_correct=arc['correct'],external_gate_open=gate)),flush=True)


if __name__=='__main__':main()
