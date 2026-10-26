# Codex Research Task: Training-Free Looped Transformers — Stage 1 Reproduction & Trajectory Audit

## 0. 任务定位与核心 Goal

请以独立研究工程的标准，自主完成一轮 **Looped Transformer Baseline Reproduction & Mechanistic Trajectory Audit**。

本轮不提出新模型、不训练 LLM，也不以提升 Accuracy 为唯一目标。

**核心研究目标：**

系统复现 Training-Free Looped Transformers、LoopCD 和 Dynamic Halting，研究额外循环计算如何影响隐藏状态、预测正确性和计算效率，判断循环是否带来真实、可识别的预测修正，以及现有方法在哪些条件下失效。

最终要回答三个问题：

1. 对普通预训练 Transformer 增加受控循环，究竟改善了什么？收益是否只是概率重分配，还是能够将错误答案纠正为正确答案？
2. 对已经训练为循环结构的 LLM，额外循环是如何改变预测的？是否存在早期正确、后期错误，或早期错误、后期纠正的现象？
3. KL、Confidence、Margin、Hidden L2 等指标能否可靠预测后续循环是否仍有价值？是否存在预测已经稳定，但停止过早会损失后续收益的情况？

本轮结果将用于决定后续是否研究 Continuation-Aware / Continuation-Preserving Latent Computation。

**不要为了支持该研究方向而筛选结果。允许结论是这个问题不显著、不具备可预测性，或者已有方法已经解决。**

---

## 1. 执行权限与资源约束

你拥有本轮实验工程的较大自主决策权。

允许：

- 阅读论文、官方代码、模型实现及相关技术文档。
- 使用网络下载模型、数据集和公开代码。
- 自主选择合理的软件版本，安装所需依赖。
- 创建实验目录、脚本、测试、日志、图表和报告。
- 在不改变核心实验协议的前提下，修复兼容性及性能问题。
- 根据实际 GPU 吞吐调整 batch size、评测批次及执行顺序。
- 在显存和计算预算允许范围内，自主补充必要的诊断对照。
- 发现方法实现错误时，修复并重新运行受影响实验。

不允许：

- 训练或微调主 LLM。
- 为追求涨点而引入新的模型模块。
- 使用测试集调节超参数。
- 在不同方法间改变输入模板、答案评分规则或样本集合。
- 将作者论文报告的结果当作自己的复现结果。
- 隐藏失败实验、负结果或没有达到预期的设置。
- 覆盖仓库中与本任务无关的已有实验。
- 在未核验代码的情况下把不同论文中的同名方法视为相同实现。

请首先检测当前硬件环境。

目标硬件：

- 优先支持单卡 RTX 4090 24GB 或 RTX 5090 32GB。
- BF16 优先。
- 使用已有 checkpoint，不进行预训练。
- 正式实验总预算预期为 20–50 GPU-hours。
- 将 50 GPU-hours 设为本轮硬性算力上限。
- 优先将实际实验和缓存控制在 60GB 磁盘空间以内。

开始时先进行小规模 benchmark，实测吞吐和显存，并重新估算完整实验的 GPU-hours。

如果预计超过预算，应优先缩小数据量，而不是删掉核心对照。

请直接执行任务，不要只输出研究计划。除非涉及凭证、不可访问的必要资源或其他无法自主解决的阻塞，否则无需等待人工确认。

---

## 2. 项目目录与参考资料

如果当前已经位于一个 Git 仓库，优先使用当前仓库。

在其中建立独立目录：

`experiments/looped_stage1_trajectory_audit_seed2021_v1/`

如果当前没有适合的仓库，则新建独立工程，不修改无关项目。

需要保留所有脚本、配置、原始结果、依赖版本和最终报告。

### 必须阅读的论文

**Paper A: Training-Free Looped Transformers**

[https://arxiv.org/abs/2605.23872](https://arxiv.org/abs/2605.23872)

公开实现：

[https://github.com/L-z-Chen/Training-Free-Looped-Transformer](https://github.com/L-z-Chen/Training-Free-Looped-Transformer)

重点理解：

- Naive Loop
- Block-mode / Layer-mode
- Damped Euler
- Anchored Euler
- 循环窗口位置
- 缓存策略
- Prefill-only 与 full decoding 的区别

**Paper B: Decoding Looped Transformers Better for (Almost) Free**

[https://arxiv.org/abs/2610.02185](https://arxiv.org/abs/2610.02185)

重点理解：

- LoopCD-Logits
- LoopCD-Hidden
- Early vs. Final recurrent states
- Contrastive decoding
- 不同循环深度的性能变化

注意：2026 年 9 月还有另一篇同名 LoopCD，不要混淆。优先按照上述 arXiv ID 的公式与实验设置实现。

**Paper C: Adaptive Depth in Looped Transformers**

[https://arxiv.org/abs/2607.20519](https://arxiv.org/abs/2607.20519)

**Paper D: Per-Token Fixed-Point Convergence**

[https://arxiv.org/abs/2607.14427](https://arxiv.org/abs/2607.14427)

重点理解：

- Per-depth prediction
- Confidence-based exit
- KL-based exit
- Hidden-state stability
- Accuracy–Compute Pareto

### 需要使用的模型

**Model A**

`Qwen/Qwen3-4B-Base`

[https://huggingface.co/Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)

用于 Training-Free Loop。

**Model B**

`ByteDance/Ouro-1.4B`

[https://huggingface.co/ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B)

用于 recurrent trajectory、LoopCD 和 dynamic halting。

可选：

`ByteDance/Ouro-1.4B-Thinking`

用于少量生成式数学推理验证；只有在预算允许、基础模型出现明显地板效应时才加入。

不要在第一轮下载或运行 30B 以上模型。

开始实验之前检查 checkpoint、代码版本及实际网络结构，不能仅根据本文预设的层数或 API 实现。

---

## 3. Phase 0：环境、协议与正确性检查

这一阶段必须首先完成。

### 3.1 环境检查

记录：

- GPU 型号、显存、驱动和 CUDA。
- PyTorch、Transformers、lm-evaluation-harness 版本。
- 模型权重 revision 或 commit SHA。
- 数据集版本和数据快照。
- GPU 峰值显存。
- 每次测试的运行时间。

所有实验固定随机种子 `2021`，必要时使用多个确定性子种子，并记录来源。

默认不启用量化。如果显存不足，可以降低 batch size、缩短上下文，优先保留 BF16 权重。

### 3.2 官方 Baseline 一致性测试

对于 Qwen3：

- 比较原始 Hugging Face forward 与未启用循环的 patched forward。
- `K=1` 时应与单次原始窗口执行一致。
- Anchored 方法在 `β=1` 时应恢复原始窗口输出。
- 记录 logits 最大绝对误差、相对误差及答案一致率。

对于 Ouro：

- 使用默认四次循环执行。
- 检查循环深度参数。
- 验证通过不同方式提取中间状态时，最终深度输出与正常模型调用一致。
- 避免因为重复施加输出归一化、coda 或 LM head 而改变结果。

使用与 BF16 数值精度相符的容差。

如果存在不符合预期的差异，优先调查 forward、position embedding、attention mask、KV-cache 和 state normalization，不应直接开始正式实验。

### 3.3 性能预检

使用 20–50 个样本测试：

- 推理吞吐。
- 峰值显存。
- 不同循环深度的实际延迟。
- 保存中间状态的内存开销。

根据预检结果给出正式运行的计算预算。

---

## 4. Phase 1：Training-Free Looped Transformer 复现

### 4.1 研究问题

普通 Transformer 能否在冻结权重的情况下，通过重复执行中间层产生新的有效预测？

如果能够，受控循环相比直接重复究竟改善了什么？

### 4.2 模型

固定：

`Qwen/Qwen3-4B-Base`

采用官方公开实现中的中间窗口：

`[15, 16, 17, 18]`

优先采用作者实现的 `block-mode`。

必须理解并复用其 KV-cache 策略，不能直接将窗口整体复制 K 次后假设等价于官方方法。

### 4.3 实验组

| ID | 方法             | 设置         |
| -- | -------------- | ---------- |
| A0 | Original Qwen3 | 不循环        |
| A1 | Naive Loop     | K=2        |
| A2 | Naive Loop     | K=6        |
| A3 | Damped Euler   | K=6        |
| A4 | Anchored Euler | K=6, β=0.5 |

其中 A4 使用论文公开的主要 recipe：

- `strategy=block_anchored`
- `loop_indices=[15,16,17,18]`
- `K=6`
- `anchor_beta=0.5`
- `cache_strategy=first`
- `decode_mode=bypass`，仅用于与原论文一致的 log-likelihood 评测

若需要生成式评测，必须独立启用并验证完整 decode 模式，不能把 prefill-only 结果当作逐 Token 循环生成结果。

不要为了获得正收益而在测试集搜索窗口、K 或 β。

### 4.4 数据集

优先使用：

- ARC-Challenge
- OpenBookQA
- MMLU Mathematics / Formal Logic 相关子集

建议规模：

- ARC-Challenge：50 验证 + 200 独立测试。
- OpenBookQA：50 验证 + 200 独立测试。
- MMLU：约 200 独立评测样本；若没有合适的验证 split，则将其作为纯 held-out 测试，不用于调参。

遵循数据集既有的官方 split。不能混合验证与测试；如果需要从同一公开 split 二次划分，必须提前用固定 seed 创建不可重叠的 manifest，并在结果中明确说明。

统一 prompt template、tokenizer、few-shot 设置以及候选答案评分规则。

优先使用论文可复现的原始 benchmark 协议。若建立自己的补充评分协议，则要将其与论文原始结果分开。

多 Token 候选需要统一处理长度归一化；不能在不同模型配置之间改变规则。

### 4.5 指标

至少报告：

- Accuracy
- NLL 或相应的条件 log-likelihood
- Correct → Wrong
- Wrong → Correct
- Correct → Correct
- Wrong → Wrong
- 平均输出熵
- Gold-answer log-probability
- 推理时间
- 理论计算量相对变化
- 峰值 GPU 显存

此外，对于循环窗口，记录：

- 输入与输出的相对 L2 变化
- Cosine similarity
- 中间状态更新幅度
- 不同循环设置的预测分布差异

要特别检查：

**循环带来的净 Accuracy 增益，是否掩盖了大量方向相反的逐样本变化？**

输出模型和样本对齐的预测记录，以便后续逐题分析。

---

## 5. Phase 2：Ouro Recurrent Trajectory Audit

### 5.1 研究问题

在真正针对循环训练过的模型中，每次循环的计算是否持续产生预测收益？

我们关注的不是简单的隐藏状态收敛，而是循环过程中发生的真实预测变化。

### 5.2 模型

使用：

`ByteDance/Ouro-1.4B`

默认最大循环次数为 4。

提取四个循环深度的结果：

`r = 1, 2, 3, 4`

若原始模型 API 不直接返回所有深度的状态，可利用官方配置或经测试的 hook 获取。

保证每个深度均使用一致的 readout 路径。

### 5.3 数据

使用与 Phase 1 重叠的一部分选择题，便于解释任务差异，但必须分别记录各模型的 protocol。

建议选择约 300 道独立测试选择题。

另加入约 100 道自动验证的短答案数学题；如基础 Ouro 在这些题上接近随机表现，可以用 Ouro-1.4B-Thinking 作为单独的补充分析，不能将其结果与 Base 版本混合。

数学题要单独报告生成式 Accuracy / Exact Match，不与选择题 Accuracy 直接求平均。

### 5.4 保存循环轨迹

对每个样本记录四次循环的：

- 最终预测
- Gold-answer probability 或 log-likelihood
- Confidence
- Entropy
- Top-1 / Top-2 margin
- 与前一循环的 KL divergence
- 与前一循环的相对 Hidden L2
- Hidden cosine similarity
- Correct / Incorrect
- 运行时间

区分：

1. Full-vocabulary KL
2. 在固定选择题候选集合上归一化后的 Choice-level KL

不得混用两种指标。

Hidden L2 优先使用相对归一化的形式，并清楚说明是最后一个有效 Token、所有有效 Token 的均值，还是其他聚合方式。

只保存后续分析需要的中间状态，避免将所有样本的全量 `[layers, tokens, hidden]` 常驻 GPU 或保存到磁盘。

### 5.5 关键诊断

计算以下模式的样本数量、比例和置信区间：

- Early Correct → Final Correct
- Early Correct → Final Wrong
- Early Wrong → Final Correct
- Early Wrong → Final Wrong
- Intermediate Correct → Later Wrong → Final Correct
- Intermediate Wrong → Later Correct → Final Wrong

分析它们与以下量之间的关系：

- 初始置信度
- 循环深度
- KL
- Hidden L2
- Question length
- Task category

同时检查是否存在：

- 置信度提高但 Accuracy 下降
- Hidden L2 很小但预测发生改变
- KL 很低却在后续循环出现重要纠错
- 后期循环使已经正确的答案变错

不要预设这些现象一定存在。

---

## 6. Phase 3：LoopCD 复现与反例分析

### 6.1 研究问题

循环早期与最终状态之间的差异，是否包含有价值的纠错方向？

还是说 LoopCD 的收益只来自输出分布的 sharpening 或 calibration？

### 6.2 实现

依据 arXiv:2610.02185                                    ，实现：

**Baseline C0：**

正常使用完整循环深度的输出。

**Baseline C1：**

使用较浅循环深度的输出。

**Method C2：LoopCD-Logits**

采用论文定义的早期与最终 logits 对比：

`z_guided = z_final + omega * (z_final - z_early)`

**Method C3：LoopCD-Hidden**

采用论文定义的隐藏状态外推：

`h_guided = h_final + omega * (h_final - h_early)`

然后通过正确的下游 readout 得到最终 logits。

两种 LoopCD 方法要分别报告，不能混为同一个方法。

此外设置：

**Control C4：**

普通温度缩放或 logit sharpening。

它用于判断 LoopCD 是否真的利用了循环的特定信息，而非只是使已有预测更尖锐。

### 6.3 参数设置

在验证集上选择少量预先固定的强度：

`omega ∈ {0, 0.25, 0.5, 1.0, 2.0}`

按照论文定义保留合理的取值约束。

参数确定后不再根据测试集改变。

先在选择题候选评分上完成低成本诊断。

如果可行，再在不超过本轮预算的条件下，选取约 100 道生成式任务验证实际逐 Token LoopCD。

生成式实验必须确认每个新 Token 都正确计算了不同循环深度的状态。

不能将 prompt 最后一个 Token 的单次分数修正等同于完整 autoregressive LoopCD。

### 6.4 必须分析

- LoopCD 的总体 Accuracy 改善。
- LoopCD 将多少错误答案纠正。
- LoopCD 将多少正确答案变错。
- 哪些循环轨迹容易从 LoopCD 获益。
- 哪些轨迹容易被过度外推破坏。
- LoopCD 是否显著优于简单温度调整。
- 是否存在早期正确、后期错误且 LoopCD 加剧错误的样本。

如果选择题评分与自由生成得出不同结论，请分别报告并解释原因。

---

## 7. Phase 4：Dynamic Halting 与计算效率分析

### 7.1 研究问题

能否在不知道正确答案的情况下，根据当前循环状态判断是否应该停止？

这里需要区分：

- 预测是否稳定。
- 当前答案是否正确。
- 后续计算是否会改善预测。

这三者不能被视为同一个变量。

### 7.2 实验方法

在 Ouro 上比较：

- Fixed Depth
- KL-based Halting
- Confidence-based Halting
- Margin-based Halting
- Hidden-L2-based Halting

使用 Phase 2 保存的完整轨迹，在离线阶段模拟每个样本的退出深度。

停止判据只能读取当前及历史循环的信息，不能使用未来循环状态或 gold answer。

所有阈值在验证集上确定，再在独立测试集评估。

### 7.3 主要输出

对于每种方法报告：

- Accuracy
- 平均退出深度
- 相对于固定四轮的理论计算减少比例
- Wrong → Correct
- Correct → Wrong
- 因提前停止而损失的后续纠错数量
- 因提前停止而避免的后续退化数量

输出：

**Accuracy–Average Depth Pareto Curve**

每个方法应根据不同验证集阈值形成一条曲线，并在相同平均深度附近进行公平比较。

需要明确区分：

1. Offline Oracle：可以事后读取正确答案或未来状态，仅作为分析上限，不能算作可部署算法。
2. Deployable Halting：只使用停止当时实际可获得的信息。
3. Full-depth Fixed Baseline：完整四次循环。

### 7.4 重要限制

只缓存四轮轨迹并模拟退出，并不意味着真正减少了运行时间。

如果没有实现实际提前停止，不得声称取得了真实 wall-clock 加速。

可以报告理论 FLOPs、平均循环深度和预计节省量。

如时间允许，可实现少量真实提前停止，并测量 batch size=1 下的 latency，但必须与离线模拟结果分开。

此外，第一轮主要采用 sample-level 的选择题停止策略。不能将其直接描述成完整的 per-token adaptive decoding。

---

## 8. Phase 5：统计分析与研究可行性判断

所有实验完成后，统一进行逐样本统计分析。

### 8.1 基本统计要求

- 同一对照必须使用完全相同的样本。
- 保存所有样本 ID 和预测结果。
- 对 Accuracy 差异进行 paired bootstrap。
- 默认使用 10,000 次 bootstrap resampling。
- 固定 bootstrap seed。
- 报告 95% confidence interval。
- 对每个数据集分别报告，并提供整体汇总。
- 不允许仅报告最好的模型配置。

如果观测提升的置信区间跨零，应明确说明证据不足。

同时报告 Correct→Wrong 和 Wrong→Correct 的比例，不应仅报告净 Accuracy。

### 8.2 四个核心图表

**Figure 1：Accuracy / NLL vs. Loop Strategy**

比较 Original、Naive Loop、Damped Euler、Anchored Euler。

**Figure 2：Prediction Transition Across Recurrent Depth**

展示 Ouro 不同循环深度之间的预测变化，以及 early-correct/final-wrong 等现象。

**Figure 3：Convergence Metrics vs. Actual Prediction Changes**

展示 KL、Hidden L2、Margin 与未来纠错、未来退化之间的关系。

**Figure 4：Accuracy–Average Recurrent Depth Pareto Frontier**

比较固定深度与不同动态停止方法。

若数据允许，再增加：

**Figure 5：LoopCD Gains and Failures**

展示 LoopCD 改善及破坏的样本占比，并检查其与早期/最终预测变化方向的关系。

### 8.3 核心假设

请将以下作为待证伪假设，而不是预设结论。

**H1：** 受控循环能够减少 Naive Loop 的预测退化，但其收益不一定等价于真实多步推理能力的增强。

**H2：** Ouro 的循环过程中同时存在纠错和错误累积，而不是 Accuracy 随深度单调增加。

**H3：** 预测分布稳定性与后续计算的真实收益存在失配。

**H4：** LoopCD 的改善依赖某些特定轨迹，并非所有样本的后期预测方向都值得外推。

**H5：** 现有动态停止策略在固定计算预算下仍存在可诊断的失败模式。

对于每个假设，回答：

1. 证据支持还是反对？
2. 哪个实验最直接检验它？
3. 效应量多大？
4. 哪些对照能够排除替代解释？
5. 是否足以支撑后续研究？

不要把统计噪声或少量案例解释成普遍机制。

---

## 9. 研究结论与 Go / No-Go 决策

本轮最终目的不是发表一篇复现报告，而是决定是否值得进入下一阶段。

### 情况 A：值得继续

如果存在稳定的预测状态失配，例如：

- 某些循环深度预测高度稳定，但后续仍出现有意义的纠错。
- 提前停止会系统性错过有效计算。
- 单纯 KL 或 Hidden L2 难以判断未来收益。
- LoopCD 能改善一部分轨迹，却系统性破坏另一类轨迹。
- 这些现象不是评分协议、样本噪声或实现差异造成的。

则进一步判断：

是否存在一个可以通过反事实干预、状态复用或更合理的计算约束解决的问题？

如果成立，提出下一阶段最多两个候选研究问题，每个包含：

- 精确定义。
- 与已有工作的区别。
- 最脆弱假设。
- 一个能直接证伪它的实验。
- 所需最小算力。

### 情况 B：不值得继续

如果发现：

- 循环几乎不产生可重复的有效纠错。
- 预测变化主要来自无意义的概率扰动。
- 现有停止方法已经充分解释并利用循环轨迹。
- 没有显著的跨任务稳定失败模式。
- 候选现象只能通过知道正确答案才能识别。

则明确给出 No-Go 结论。

不要为了继续研究而人为制造新的架构复杂度。

### 特别注意

本轮不能直接宣称已经证明了“隐藏状态不可复用”。

这里的实验主要检验预测稳定性、后续计算收益和停止机制。

只有在后续加入针对中间状态的 freeze、intervention、restore 等因果实验后，才能更严格地研究 Continuation Readiness。

---

## 10. 目录与最终交付物

至少建立：

- `README.md`：项目目的和运行方式。
- `configs/`：各实验的完整配置。
- `scripts/`：下载、评测、分析、绘图和复现脚本。
- `tests/`：forward parity、loop identity、cache、scoring、alignment 等检查。
- `results/`：每个实验的原始结果。
- `trajectories/`：逐样本循环轨迹数据。
- `figures/`：实验图表。
- `logs/`：运行日志、时间及显存记录。
- `FINDINGS.md`：最终研究结论。
- `REPRODUCE.md`：从零复现说明。

结果至少包含：

1. 所有实验的完整指标表。
2. 每组配置的精确参数。
3. 每个样本的 ID、gold answer、各方法 prediction 和必要的轨迹指标。
4. 运行时间、GPU 峰值显存和 GPU-hours。
5. 统计置信区间。
6. 失败样本和反例分析。
7. 尚未完成或执行失败的项目。
8. 最终 Go / No-Go 决策。

保存原始结果为 JSONL 或 Parquet，并提供可由原始结果一键重建表格和图表的分析脚本。

不要只输出经过人工筛选的 Markdown 结果。

---

## 11. 执行策略与中间检查点

按以下顺序自主执行。

**Checkpoint 1：环境与代码正确性**

完成模型载入、官方 baseline 对齐、forward parity 和 GPU benchmark。

**Checkpoint 2：低成本 Pilot**

使用约 50–100 道样本运行所有核心配置，确认没有严重的实现问题或模型地板效应。

**Checkpoint 3：正式实验**

固定协议后扩大数据规模，执行 A、B、C、D 四组实验。

**Checkpoint 4：研究诊断**

提取错误修正、错误累积、预测稳定性失配、LoopCD 失败和早停损失。

**Checkpoint 5：最终报告**

完成全部图表、统计、失败案例和研究判断。

如果发现某个实验耗时显著超过预算，自主减少非核心扩展实验；优先保证对照完整、样本对齐和统计正确。

若提前耗尽 50 GPU-hours，停止新的 GPU 运行，基于已完成结果做分析，并明确列出缺失的实验及其影响。

不要在预算耗尽时继续运行超额任务。

---

## 12. FINDINGS.md 必须回答的最终问题

请采用研究论文的分析方式组织最终报告，而不是单纯汇报终端运行状态。

报告至少回答：

**Q1.** Training-Free Looped Transformers 在我们选定的 benchmark 上是否复现了正收益？如果没有，可能是什么原因？

**Q2.** Naive Loop 与 Damped/Anchored Loop 的主要差别体现在预测正确性、隐藏状态变化，还是输出置信度？

**Q3.** Ouro 的循环深度增加时，Accuracy 是否单调改善？不同样本的轨迹有何差异？

**Q4.** 是否存在具有代表性的 Early Correct → Final Wrong 或 Early Wrong → Final Correct？

**Q5.** KL、Margin、Confidence、Hidden L2 能否预测后续循环的真实纠错收益？

**Q6.** LoopCD 的收益来自哪些轨迹？什么情况下会强化错误？

**Q7.** 在相同平均计算量下，哪些动态停止策略最有效？哪些失败是现有方法尚未解决的？

**Q8.** 我们观察到的问题是否已经被最近的论文解释或解决？需要对相关文献进行补充检索。

**Q9.** 哪些现象只是偶然、任务特定或评分方式造成的，不能上升为通用研究问题？

**Q10.** 是否存在足以支持下一阶段研究的可重复失败模式？如果有，最小的下一轮因果干预实验是什么？

**Q11.** 下一阶段最值得研究的问题是什么？为什么比单纯改进 LoopCD、积分器或停止阈值更有价值？

**Q12.** 最终给出 Go / No-Go，并说明决定依赖的具体证据。

---

## 13. 最终执行要求

请立即开始：

1. 检查环境和当前仓库。
2. 阅读核心论文与代码。
3. 建立可复现项目。
4. 完成小规模正确性测试。
5. 估算 GPU 预算并执行正式实验。
6. 自主分析结果，修复必要问题。
7. 完成 FINDINGS.md 和全部交付物。

研究优先级：

**方法正确性 > 严格对照 > 可证伪性 > 可复现性 > 覆盖更多模型和数据集 > 单纯追求 Accuracy 提升。**

本轮最重要的交付物不是一个新的高分模型，而是一组足够可靠的实验事实，让我们能够判断 Looped Transformers 的下一步研究究竟应当解决什么问题。

不要仅仅给我一个实验计划。请实际运行、记录完整结果，并根据实验事实作出研究判断。
