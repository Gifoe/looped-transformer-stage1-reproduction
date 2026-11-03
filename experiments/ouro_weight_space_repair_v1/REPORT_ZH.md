# Ouro 免训练权重修复：实验与停止结论

本轮没有得到满足因果门槛的共享有害权重分量，主权重编辑为 **None**。原 checkpoint 保持原样，实际改变参数 **0 / 1,434,652,673 = 0%**。这不是一个“提升后的模型”。正式测试已完成全部 **1,172 题**；原生 K4→K5 纠错 17 题、损坏 39 题，净变化 -22 题。所有测试比较均在方法冻结后运行。

## 1. 问题、范围与数据隔离

目标是检验额外循环中的损害能否被定位为共享、可干预并可迁移到权重空间的计算分量，而不是预设 H2 必然成立。模型为 ByteDance/Ouro-1.4B，revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`；实际硬件按用户指定使用 GB10，替代附件的 RTX5090。未训练模型、probe 或控制器，未使用梯度、反向传播、优化器或 LoRA。

ARC 数据版本为 `210d026faf9955653af8916fad021475a3f00453`。从1119条官方 train 按 `random.Random(2021).sample` 固定25条示例，排除示例及问题文本重复项后有1093条可用；确定性 SHA 排序后取 discovery192、selection192，互不重叠。官方 validation299只在候选定义冻结后读取；官方 test1172只在 `FINAL_LOCK.json` 后下载、构造与评估。不以四选项过滤官方测试集。旧测试标签、逐题预测和轨迹未用于假设生成。

用户明确选择“固定同一组25例，重跑全部对照”。旧流程逐题推进的示例采样器会在1172次抽样中覆盖全部1119条 train，无法留下独立 discovery。新协议解决数据隔离冲突，但改变历史逐题 prompt；本报告不声称与旧 varying-shot 数字逐点相同。所有本轮方法使用同一固定 prompt。测试集曾在早期工作使用，因此称为锁定回归基准，不称全新盲测。

## 2. 评分与实现正确性

使用 BF16、lm-eval0.4.9、seed2021；对每个完整候选续写计算所有目标 token 的条件 log-likelihood，除以候选字符串字符数，最大者作为 `acc_norm` 预测。margin 为正确选项归一化分数减去最高错误选项分数。它不是校准的选项概率。K1/K2仅作辅助轨迹与 LoopCD 参考，主要深度曲线为 K3–K6。

每个答案候选只运行一次已验证的K6完整原生轨迹，并从同一轨迹读取各循环输出，计算LoopCD与固定插值读出；这些统计输出不是11次独立的transformer前向。独立深度一致性检查用于验证中间状态可复用，不是独立K3/K4/K5速度基准。报告中的GPU账本是共同加载模型、前向与读出分析所在worker的合计耗时，不能分摊后宣称单一深度的推理延迟。

正式前修正了一个实现问题：显式全1 attention mask 与旧脚本省略 mask 在 HF4.55.4 中触发不同 SDPA/GQA 路径，确实改变少量 BF16 预测。旧 `discovery_audit`、部分 `layer_screen_v1` 已隔离为工程记录，不进入任何正式表格。正式路径精确恢复为 `model.model(input_ids=x,use_cache=False)`。缓存优化也未通过原定 BF16 一致性门槛，未用于正式评估，没有放宽容差。

3道真实25-shot训练题的原始调用/新评分器候选分数、逐 token 概率、隐藏状态，以及独立K3/K4/K5与K6中间输出，误差全部为0；原始重复运行也相同。独立 HFLM K4 检查通过原有每token0.05容差，实际最大误差0.002395；这与对原直接调用的零误差核对是两项不同检查。完整代表题记录钩子也通过精确 no-op；边界干预的K5分数与原生状态插值读出逐候选精确相等，再以K6检查后续利用。证据分别保存在 `results/original_call_parity/`、`results/discovery_original/PARITY.json`、`results/layer_screen_original/NOOP_GATE_SUMMARY.json` 和 `BOUNDARY_READOUT_IDENTITY.json`。

## 3. 原生深度曲线

| 划分 | K | 正确数 | acc_norm | 平均正确答案margin |
| --- | --- | --- | --- | --- |
| discovery | 3 | 119/192 | 61.979% | 0.063078 |
| discovery | 4 | 118/192 | 61.458% | 0.063375 |
| discovery | 5 | 120/192 | 62.500% | 0.052379 |
| discovery | 6 | 118/192 | 61.458% | 0.055322 |
| selection | 3 | 111/192 | 57.812% | 0.073747 |
| selection | 4 | 114/192 | 59.375% | 0.082981 |
| selection | 5 | 110/192 | 57.292% | 0.069898 |
| selection | 6 | 111/192 | 57.812% | 0.073875 |
| validation | 3 | 165/299 | 55.184% | 0.038167 |
| validation | 4 | 162/299 | 54.181% | 0.037243 |
| validation | 5 | 161/299 | 53.846% | 0.025610 |
| validation | 6 | 163/299 | 54.515% | 0.025737 |
| test | 3 | 696/1172 | 59.386% | 0.062737 |
| test | 4 | 702/1172 | 59.898% | 0.065830 |
| test | 5 | 680/1172 | 58.020% | 0.056714 |
| test | 6 | 680/1172 | 58.020% | 0.055821 |

Discovery 中 K4→K5 有 5 题纠错、3 题损坏，净增2题；这个小样本结果不能替代完整测试集。**完整测试确实发生额外循环退化**：K4→K5 纠错 17 题、损坏 39 题，下降 -1.877 pp；95%配对bootstrap区间为 [-3.157, -0.683] pp，精确McNemar p=0.00456，划分内Holm p=0.02281。这些仍是冻结的次要比较，不是权重编辑的主检验。

K5与K6的准确率相同，但逐题有 21 次纠错、21 次损坏，不能把准确率持平解释为答案不再变化。平均margin与离散准确率也不能相互代替。各划分独立报告，三道训练损坏题与完整测试的39道损坏题不混作同一机制样本。

## 4. H1 / H2 / H3 与失败案例

对归一化循环边界状态定义 `h(λ)=h4+λ(h5−h4)`，λ固定为0、0.125、0.25、0.5、1。它首先是读出路径；另测 raw pre-boundary 状态插值后只调用一次原 final norm，二者分开记录。预定有限步超调判据要求小正步改善margin、完整步变差，且至少一个内部点优于两端，epsilon=0.001。

三道正式 K4→K5 损坏题中，2道在内部网格恢复正确，但 **0道**满足预定超调判据。因而不能把插值恢复直接写成已证明 H1。未恢复网格的案例也不能直接归入H2：λ=0本来正确，更小正步仍可能恢复；H3也没有被证明，混合早期状态并不说明原生h5保留了全部所需信息。

| 训练题ID | K4margin | K5margin | 最佳内部margin | 内部点恢复 |
| --- | --- | --- | --- | --- |
| Mercury_7008050 | 0.006165 | -0.013019 | 0.003556 | 是 |
| MCAS_2013_5_29397 | 0.001446 | -0.006938 | 0.000702 | 是 |
| Mercury_7180758 | 0.001562 | -0.017957 | -0.001610 | 否 |

完整题面、选项、标签、所有原生与插值结果保存为 `failure_cases.jsonl`；逐题原生循环变化与分类见 `failure_mechanisms.csv`。这些都是训练 discovery 案例，未在测试后挑选新假设。

## 5. 逐层干预与共享调用检查

按冻结的 SHA 规则选2道损坏、2道纠错、2道稳定正确、2道稳定错误题。正式筛查24个物理层×attention/MLP，共48个模块；每次仅在第5循环将某分支的 **post-branch-RMSNorm residual write** 乘0.5。加上原始记录与2种边界控制，共51条件×8题=408次逐题评分。干预作用于全序列所有位置，记录只保留每个候选的全部预测位置。物理层编号从0开始，循环编号从1开始。

同时记录 raw projection output、post-norm write、循环间变化、L2/相对范数、cosine及与实际边界位移的方向对齐。它们是描述性几何，不是可加因果归因：分支和最终RMSNorm会改变后续方向与尺度。原始紧凑BF16向量保存在各 `trace_*.pt`，汇总与方向CSV在筛查目录。

边界控制把第5轮归一化输出按固定0.25或0.5步与该轮入口混合，再送入第6轮。其K5等于对应读出插值，K6则测进一步计算是否保留变化。局部模块的资源门槛要求净纠错为正、不能损坏代表性原生纠错题或稳定正确题，并超过已测边界控制；这是探索性门槛，不是假设检验。

| 探索模块层 | 分支 | K5净纠错 | 平均Δmargin | 后续资格 |
| --- | --- | --- | --- | --- |
| 0 | mlp | 1 | 0.026738 | localization_only_requires_shared_invocation_check |
| 10 | attention | 1 | 0.007818 | localization_only_requires_shared_invocation_check |

48个模块中有6个达到局部门槛，先按该门槛筛选再排序；仅对冻结短名单中的2个模块追加了共享调用确认，并未全循环测试其余局部合格模块。局部与全循环结果如下；样本来自同一组 discovery 代表题，不是独立泛化估计。

| 层(0起) | 分支 | 作用范围 | K | n | 纠错 | 损坏 | 净纠错 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | mlp | loop5 | 3 | 8 | 0 | 0 | 0 |
| 0 | mlp | loop5 | 4 | 8 | 0 | 0 | 0 |
| 0 | mlp | loop5 | 5 | 8 | 1 | 0 | 1 |
| 0 | mlp | loop5 | 6 | 8 | 0 | 2 | -2 |
| 0 | mlp | all_loops | 3 | 8 | 1 | 0 | 1 |
| 0 | mlp | all_loops | 4 | 8 | 1 | 1 | 0 |
| 0 | mlp | all_loops | 5 | 8 | 1 | 1 | 0 |
| 0 | mlp | all_loops | 6 | 8 | 0 | 2 | -2 |
| 10 | attention | loop5 | 3 | 8 | 0 | 0 | 0 |
| 10 | attention | loop5 | 4 | 8 | 0 | 0 | 0 |
| 10 | attention | loop5 | 5 | 8 | 1 | 0 | 1 |
| 10 | attention | loop5 | 6 | 8 | 0 | 1 | -1 |
| 10 | attention | all_loops | 3 | 8 | 0 | 2 | -2 |
| 10 | attention | all_loops | 4 | 8 | 0 | 2 | -2 |
| 10 | attention | all_loops | 5 | 8 | 0 | 2 | -2 |
| 10 | attention | all_loops | 6 | 8 | 0 | 2 | -2 |

最强的继续理由也必须保留：唯一在既定插值网格未恢复的 `Mercury_7180758`，经第0层MLP全循环阻尼后在K5恢复正确，margin从 -0.017957 升至 0.116709；同题K3也被纠正，K4/K6保持正确。这是一个真实、跨调用保留的单题因果线索，不应被汇总值掩盖。

但同一MLP干预在代表题K5只有1次纠错和1次回退，净收益0；K6有2次回退、0次纠错，并损害另一道原生纠错题。注意力第10层全循环干预在K3/K4/K5/K6均是0次纠错、2次回退。局部第5轮干预本身也没有稳定第6轮：MLP和attention分别损失2题和1题。它们表明模块执行具有题目与循环深度依赖性，尚未隔离一个可删除的共享有害方向。

48模块中挑出的最佳结果存在选择偏差，8道代表题不能作为独立泛化集。因此不报告这些排序的显著性，也不把单题修复称为发现通用有害方向。分支标量阻尼定位的是执行敏感性，不是输出矩阵投影：Ouro先执行o_proj/down_proj，再做分支RMSNorm；对矩阵整体正比例缩小会近似被归一化抵消。只干预第5轮也不等于共享物理权重编辑。

## 6. 权重阶段决定

48模块筛查后只确认了按规则选出的两个模块。MLP物理第0层在第5轮能修复一题，但全循环K5为1次纠错抵消1次损坏，K6损坏2题；Attention物理第10层全循环在K3/K4/K5/K6均损坏2题、无纠错。最强支持继续的案例是Mercury_7180758：MLP0全循环确实恢复它，K5margin由−0.017957升至0.116709；但这同时破坏其他原生纠错，尚无可分离方向及独立问题迁移证据。两个已确认模块均未通过共享调用门槛，因此停止本轮H2→方向搜索→权重编辑路线，不据此否定一切投影修复。

停止范围限定为本轮48模块筛查后选出的两条局部线索及其共享调用门槛。全分支标量阻尼失败不能逻辑推出所有更窄的投影方向都会失败；其含义是现有证据不足以支持继续增加方向搜索，而非H2已被普遍否定。

PCA/SVD方向诊断、方向特异性对照和实际rank1/rank4权重编辑因推进门槛未通过而**未开展**，不能写成这些实验已开展且结果为负。未产生监督拟合方向。`weight_edit_manifest.json`明确记录None、原权重哈希和0个实际BF16改动坐标。因此不存在“编辑后K5”的主对比；该对比为不可用，不是零提升实验。没有候选可供范数匹配时，随机权重编辑对照条件性不适用，不能声称它已失败或被击败。所有已运行激活对照完整公开。

## 7. 冻结的1172题回归评估

候选门槛、None决定、原生深度、原始LoopCD公式与3个固定插值系数在 held-out 评估前固定。Selection、validation不用于生成新方向。官方validation结束后冻结最终方法清单，再读取test。没有根据测试选择最佳K或λ。Fixed LoopCD为 `z4+0.5(z4−z1)`；Adaptive为逐token `z4+[1−(pmax−psecond)](z4−z1)`，概率来自K4完整词表。

| 冻结方法 | 正确数 | acc_norm | 平均正确答案margin |
| --- | --- | --- | --- |
| native_k4 | 702/1172 | 59.898% | 0.065830 |
| native_k5 | 680/1172 | 58.020% | 0.056714 |
| loopcd_k4_fixed | 717/1172 | 61.177% | 0.099067 |
| loopcd_k4_adaptive | 716/1172 | 61.092% | 0.096911 |
| normalized_chord_0.125 | 703/1172 | 59.983% | 0.064804 |
| normalized_chord_0.25 | 704/1172 | 60.068% | 0.063979 |
| normalized_chord_0.5 | 696/1172 | 59.386% | 0.061957 |

本轮固定25示例协议下，Fixed LoopCD纠错 41 题、损坏 26 题，净增 15 题；Adaptive分别为 52 与 38，净增 14 题。两者相对K4的点估计为正，但本轮配对区间均包含0，不能宣称显著改善。固定插值系数全部如实报告；其对K5的恢复不能被写成共享权重修复，也没有根据测试挑选其中一个作为主方法。

| 测试配对比较 | 纠错 | 损坏 | Δaccuracy(pp) | 95%配对bootstrap CI(pp) | 精确McNemar p | 划分内Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| native_k3 → native_k4 | 30 | 24 | +0.512 | [-0.683, +1.706] | 0.4966 | 0.9932 |
| native_k4 → native_k5 | 17 | 39 | -1.877 | [-3.157, -0.683] | 0.004562 | 0.02281 |
| native_k5 → native_k6 | 21 | 21 | +0.000 | [-1.109, +1.109] | 1 | 1 |
| native_k4 → loopcd_k4_fixed | 41 | 26 | +1.280 | [-0.085, +2.645] | 0.08644 | 0.3457 |
| native_k4 → loopcd_k4_adaptive | 52 | 38 | +1.195 | [-0.341, +2.730] | 0.1702 | 0.5107 |
| native_k5 → normalized_chord_0.125 | 34 | 11 | +1.962 | [+0.853, +3.072] | 0.0008241 | 0.005769 |
| native_k5 → normalized_chord_0.25 | 31 | 7 | +2.048 | [+1.024, +3.072] | 0.0001162 | 0.0009293 |
| native_k5 → normalized_chord_0.5 | 20 | 4 | +1.365 | [+0.597, +2.218] | 0.001544 | 0.009263 |

Bootstrap以问题为单位配对重采样，10000次、seed2021；区间为每个比较的95%百分位区间，并非多重比较调整后的同时置信区间。McNemar为不一致对数上的双侧精确二项检验；Holm仅调整各划分内列出的次要检验族，族大小在CSV中记录，不调整bootstrap区间，也不宣称跨所有划分共同控制错误率。所有这些均为无主编辑情况下的次要比较，不能挑出最好一行当主方法。Discovery统计尤其属于探索性描述。`paired_statistics.csv`还提供margin差和区间；`paired_transitions.csv`保留逐题预测、分数组、纠错与损坏。

## 8. 对五个研究问题的回答

- **Q1：额外循环为何损失表现？** 完整测试明确观测到K4→K5净损失22题（−1.877pp）；它不是“没有退化”。正式discovery只有三道局部损坏，且净增2题，显示小样本机制筛查不能替代总体评估。现有干预证明了特定案例对读出/分支与干预调用范围的敏感性，但没有唯一识别完整测试退化的原因。三道训练题不构成39道测试损坏题的机制解释；测试后没有据这些错误继续挑选模块、方向或系数。
- **Q2：有共享有害权重分量吗？** 本轮未建立。隐藏状态几何、最佳局部模块及标量消融不满足共享方向的因果与迁移要求。这不是不存在此类分量的证明。
- **Q3：Astra能无训练移除它吗？** Astra完成独立架构审查、干预实现与停止审查；证据未达到提出可信权重编辑的门槛，因此没有移除结果。不能用审查能力替代实验成功。
- **Q4：是否改善未见题和额外循环？** 没有可声明的权重修改收益。Selection/validation/test给出的是冻结基线与读出控制的完整曲线，而不是编辑模型泛化成功。
- **Q5：是否值得继续作为可发表方向？** 当前证据不足，结论为本轮权重修复路径No-Go。已有插值恢复、层消融和BF16路径差异本身不足以构成所要求的新机制。重新投入的必要条件是独立未见题上的方向特异效应、范数匹配随机/标量控制，以及实际共享BF16权重的迁移；本报告不把尚未完成的条件写成发现。

## 9. 资源、产物和复现

独立8小时账本累计 **1.6395 GPU-worker hours**（5902.22秒），包括模型加载、失败检查、被替代的mask实验及正式运行。下载和纯CPU准备不冒充GPU计算。未启动训练，也未改动旧实验的预算。各次receipt在 `logs/gpu_budget.json`；原模型文件SHA256为 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`。

本地目录 `D:\chenyu-iclr\experiments\ouro_weight_space_repair_v1`；服务器目录 `/home/newuser1/experiments/ouro_weight_space_repair_v1`。环境版本见 `logs/ENVIRONMENT.json`；Torch/CUDA通过只读共享环境复用，Transformers4.55.4等覆盖包位于本实验独立env。复建可用 `OURO_SHARED_ENV` 指向兼容的Torch环境，再执行bootstrap。不要在已有结果目录覆盖重跑。

主要入口：`check_original_call_parity.py` → `run_discovery.py --uncached --study discovery_original` → `analyze_stage_a.py` → `run_layer_screen.py` → `analyze_layer_screen.py` → 条件性 `run_shared_confirmation.py`。所有GPU任务经 `supervise.py` 限时与记账。冻结阶段入口为 `run_no_edit_evaluation.py`，拒绝缺失的决定或lock；测试下载亦有强制lock与固定LFS哈希检查。统计入口为 `analyze_evaluation.py --include-test`。

八类交付均保存：本报告、failure_mechanisms、layer_intervention_results、weight_edit_manifest、evaluation_summary、paired_transitions、完整脚本和图表。另附failure_cases、paired_statistics、严格门槛与完整原始向量/日志。图表见 `figures/accuracy_and_transitions.png` 及筛查目录的分支/边界/方向图。

## 10. 研究来源

- [MIT Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) 与 [Experiment8日志](https://github.com/RikiyaT/lm_understand-logs)：为条件性权重编程提供动机，不保证不同模型与错误类型可迁移。
- [Ouro项目](https://ouro-llm.github.io/)：共享堆栈循环；实际分支与边界归一化以固定checkpoint源码为准。
- [Right Direction, Wrong Step](https://arxiv.org/abs/2609.16665)：插值/步长现象已有研究，不能将本轮插值恢复本身声称为新颖机制。
- [原复现仓库](https://github.com/Gifoe/looped-transformer-stage1-reproduction)、[固定版本harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9)、[ARC数据](https://huggingface.co/datasets/allenai/ai2_arc)。固定版本、源文件与哈希在sources和configs中留存。
