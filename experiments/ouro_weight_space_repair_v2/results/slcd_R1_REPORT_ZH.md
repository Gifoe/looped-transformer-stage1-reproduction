# Training-free LoopCD 自主迭代实验报告

本轮在原V2文件夹内完成，所有新文件使用slcd_前缀；不覆盖原V2报告、结果或权重。模型为原始Ouro-1.4B，未训练、未编辑权重。最终主方法 `A1_g1p0`：**726/1172（61.945%）**，Fixed为717/1172，净+9题（+0.768pp），超过717的执行门槛，没有获得可靠的配对统计支持。

## Q1. 原始LoopCD为什么产生有害改答？

Fixed把早期到K4的logit差继续外推，但该方向并非每个候选或样本都与正确答案一致。训练discovery384中，相对K4有10次纠错、14次退化；selection192中为4/4。候选分差及跨循环一致性是本轮检验的退化识别信号，其稳定性由确认训练集结果衡量。这个实验支持“统一外推会改变部分已正确排序”的操作性解释，不能据此断言某种隐藏状态因果机制。对gold margin的改善也不等同于top1正确数改善。

## Q2. 实现了哪些方法？

共注册47种配置（含基线与对照），研究5轮：A1字符归一化完整候选分数的晚期修正；A2全词表token logits修正后重新log-softmax；相对最终logits的正交/平行分解，centered和raw对照；确定性候选门控；基于失败模式的最小混合；gamma局部邻域、guidance强度和参考循环改变。没有训练门控器、拟合分类器或改动模型权重。A1与A2不等价。全部实验参数在configs/slcd_METHODS.json中。

正交投影采用a=z4、d=z4-z1、p=a(d·a)/(a·a)、o=d-p；centered版本先减各自词表均值，以获得softmax平移不变性；raw版本不中心化。原论文§5.2/F.2没有明确规定中心化，不能把这里的centered约定称为作者精确实现。[原论文](https://arxiv.org/html/2610.02185v1#S5.SS2)。零范数时p=0，无epsilon ridge。

## Q3. 自主迭代发现了什么？

初始A1和A2小系数的训练收益促使继续分析退化。训练特征提示K4/K5一致性及Fixed候选分差可能保护原答案，于是引入单一门控。确认训练集中，已评估的3个混合规则分别有0个正净收益、0个持平、3个负净收益。确认训练集简单方法结果：Ref2_w0p5净+2题；A1_g1p0净+4题。正交/平行引导的两训练子集结果完整保留在消融表，未把单一子集的上升当作稳健优势。没有根据已暴露的历史test726选择gamma，也没有在test后修改参数。

| method | discovery | selection | confirmation |
| --- | --- | --- | --- |
| fixed | 235/384 (+0) | 114/192 (+0) | 333/517 (+0) |
| H_A1_g1p0_cd_agrees_k5 | 242/384 (+7) | 117/192 (+3) | 330/517 (-3) |
| H_A1_g1p0_protect0p025 | 246/384 (+11) | 116/192 (+2) | 332/517 (-1) |
| H_A2_g0p25_cd_agrees_k5 | 246/384 (+11) | 116/192 (+2) | 327/517 (-6) |
| A2_g1p0 | 239/384 (+4) | 116/192 (+2) | 332/517 (-1) |
| Ref2_w0p5 | 238/384 (+3) | 116/192 (+2) | 335/517 (+2) |
| C_fixed_low_revert_t0p025 | 242/384 (+7) | 115/192 (+1) | 329/517 (-4) |
| A2_g0p25 | 240/384 (+5) | 115/192 (+1) | 332/517 (-1) |
| A1_g1p0 | 237/384 (+2) | 116/192 (+2) | 337/517 (+4) |


括号为相对Fixed的净纠错。517题在反馈后参与晋级，属于选择数据。前3晋级规则在结果揭示前写入slcd_ADVANCEMENT_POLICY.json。

## Q4. 开发数据上最佳方法与最终选择是什么？

旧discovery上最高的混合规则不等于最可信方法。确认训练集按预定排序晋级 `A1_g1p0, Ref2_w0p5, A2_g0p25`；随后只进行一次官方validation299比较，按正确数、三训练子集最差比例净收益、较少的纠错与退化总数、逻辑深度、复杂度和ID顺序决定主方法。Fixed/Adaptive只作比较，不在新方法主候选中；没有secondary。

| method | correct | n | net | R | D |
| --- | --- | --- | --- | --- | --- |
| native4 | 162 | 299 | -6 | 7 | 13 |
| fixed | 168 | 299 | 0 | 0 | 0 |
| adaptive | 172 | 299 | 4 | 9 | 5 |
| A1_g1p0 | 173 | 299 | 5 | 7 | 2 |
| Ref2_w0p5 | 170 | 299 | 2 | 12 | 10 |
| A2_g0p25 | 169 | 299 | 1 | 2 | 1 |


最终算法及所有依赖：
```json
{
  "id": "A1_g1p0",
  "kind": "score_late",
  "gamma": 1.0,
  "late": 5,
  "required_depth": 5,
  "complexity": 1
}
```
完整递归定义、数值系数、source/model/tokenizer/data/prompt hashes及测试授权见根目录FINAL_METHOD_LOCK.json。该锁在访问本轮测试分数前创建。

## Q5. 完整ARC是否超过717题？

| method | correct | n | accuracy_percent | net |
| --- | --- | --- | --- | --- |
| native4 | 702 | 1172 | 59.898 | -15 |
| fixed | 717 | 1172 | 61.177 | 0 |
| adaptive | 716 | 1172 | 61.092 | -1 |
| A1_g1p0 | 726 | 1172 | 61.945 | 9 |


全部1172题，固定同一组25-shot，seed2021，原始BF16 checkpoint，lm-eval0.4.9完整候选条件似然，按答案字符数归一化、首个argmax打破平局。复用兼容的1172题原始完整候选分数，CPU执行冻结公式；本轮test GPU重放覆盖为0。A1的线性修正是排名分数，不能解释为校准的token对数概率。模型revision：574fa66cb8bf5abdc979642d01cf2b79b16bfab1。

## Q6. 纠错和新增错误各有多少？

主方法相对Fixed：纠错18题，退化9题，净+9题；全部改答46题。错误到错误也计入改答，不能混入纠错计数。逐题完整转移及分差见slcd_paired_transitions.csv；全部原始候选分数与提示在记录归档。

## Q7. 统计和计算上是否值得？

主比较为唯一冻结主方法vsFixed。配对50,000次题级bootstrap的95%差值区间为[-0.085, 1.621]pp；双侧exact McNemar p=0.122078。超过717只是后续benchmark执行条件，并不等同于显著优越。训练/validation经过选择，不把其未校正p值作为确认性证据；次要比较未作多重显著性声明。

本轮GPU-worker累计2206.406秒（0.6129小时），包括所有实际worker和失败尝试，不混入旧V1/V2预算。预检失败但未启动GPU的命令不收费。主方法数学所需深度5，而本实现统一实际执行K6，额外头读取也有成本。源码门控公共接口还会访问K5；不能把数学深度当已实现的K4部署。没有独立生产路径测速，禁止声称共享轨迹评估带来延迟改善。

## Q8. 是否跨benchmark泛化？

ARC主方法通过执行门槛，但没有外部任务的完整结果；不能据此判断跨任务效果。

外部任务未完成/未运行状态（不报告部分样本准确率）：
| task | status | reason |
| --- | --- | --- |
| hellaswag | incomplete | supervised worker failed or exhausted its budget; no partial accuracy reported |
| openbookqa | incomplete | supervised worker failed or exhausted its budget; no partial accuracy reported |

其他任务只有完整ARC主方法通过门槛后才获授权；如运行，均使用lm-eval0.4.9官方任务定义、固定0-shot和acc_norm，四种方法共享同一prompt，算法及系数保持冻结。这与ARC25-shot不同，不能直接跨任务比较绝对准确率。新任务无独立历史参考，全量同轨迹公平比较不被包装成独立baseline复现。

## Q9. 是否继续？

存在正向点估计，但是否值得扩大需结合配对区间和外部任务结果；不能把一次小幅回归集提高解释为普遍算法改进。

当前研究的可交付价值是完整分数复用、token/score修正区分、精确对照和被否定的迁移假设。是否存在可推广方法需要独立未用数据的后续验证；本轮不将新模型、Loop设计或大规模训练加入范围。

## 限制、完整性和复现

- ARC test以及官方validation在历史任务中已经使用，均不能称新的盲测。历史A1 gamma1的726已暴露，是先验污染；本轮方法选择仅使用训练/validation证据，但无法消除研究者已知aggregate的影响。
- 训练按问题文本去重并排除全部25例；三个训练子集相互不重叠。官方数据跨集合文本重复审计在最终冻结后运行，见slcd_SPLIT_OVERLAP_AUDIT.json，不事后删测试题。计数：`{"confirmation__discovery": 0, "confirmation__selection": 0, "confirmation__test": 3, "confirmation__validation": 0, "demos__confirmation": 0, "demos__discovery": 0, "demos__selection": 0, "demos__test": 0, "demos__validation": 0, "discovery__selection": 0, "discovery__test": 3, "discovery__validation": 0, "selection__test": 0, "selection__validation": 0, "test__validation": 2}`。
- 之前671个文件的size/mtime和完整模型字节SHA验证通过；未改动旧实验。保留旧报告文件名，新交付使用slcd_前缀以满足同一文件夹约束。
- 初始worker收据缺口由独立CPU source/state/target审计补齐；绑定时间明确标注为事后。后续v2 worker要求完整cache授权及held-out hash锁。confirmation没有旧独立参考，只有same-forward replay，不能声称独立parity。训练576与validation299的8种基线均精确相等；test重放覆盖见BASELINE_REUSE_AUDIT。
- slcd_records.zip包含全量新提示、逐候选分数、协议、数据hash、日志和可恢复记录；体积较大的原始hidden-state .pt保存在服务器同一results目录，索引包含每个文件SHA；它们是加速缓存，不是复现必需的权重。原始模型从固定revision取得，仓库不复制模型权重。
- 数据来源与许可沿用data/DATA_SOURCE_NOTICE.txt；额外任务来源commit、raw SHA及官方task source SHA在各PROTOCOL收据中。所有新图由实际表格生成。
