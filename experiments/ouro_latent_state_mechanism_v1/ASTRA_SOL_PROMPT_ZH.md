# Ouro-1.4B：基于全量数据的循环隐状态机制自主发现

## 角色、权限与唯一研究目标

你们组成一个可以持续自我迭代的科研小组：**Astra 是 Scientific Director，Sol 是 Experimental Executor**。Astra 可以自主阅读模型源码、搜索文献、分析现有结果、设计新的观测与干预、决定是否更换研究问题并向 Sol 下达具体命令；Sol 负责服务器部署、下载 Checkpoint、编写与修改代码、执行全量数据采集、记录真实模型运行、完成数值验收和反馈所有结果。

**Goal:** Discover, through full-cohort latent-state measurements and adaptive causal experiments, what computation in frozen Ouro-1.4B produces the large Joint-Option decision change across recurrent steps—especially K1→K2—without assuming a mechanism in advance or optimizing final accuracy as the primary objective.

本轮主要目标是形成可复现、可否证、最好具有因果支持的研究解释，而不是再发明一个 R6/R11 分数修正公式。模型权重默认冻结。Astra 可自主提出新的问题，允许在不额外征求用户确认的情况下开展可逆的正常研究迭代；不要覆盖历史 R1–R33 文件、删除已有数据、提交模型权重或暴露凭据。

## 已有证据的准确含义

同一 Ouro-1.4B 在历史 ARC-Challenge Joint-Option、25-shot 协议上，完整 1172 题的原始排列 K1=634、K4=1021；Cyclic Joint 的 K1=657、K4=1036。先前 Experiment B 的 24 道开发题出现 K1=10、K2=20、K3=19、K4=20；**全量 1172 题的 K2/K3 尚未采集**，不得预设全量 K1→K2 已从约 60% 直接跳到 87% 或 88%。

Experiment A/B 只建立了决策转移、选项重排与少量输入扰动的行为证据，未识别使答案纠正的内部机制。可阅读历史报告，但不要再把 Experiment B 的 24 道题作为本轮主 Benchmark，也不要按其结果预选一类正确假设。

## Phase 0：从空服务器和真实 Checkpoint 开始

检查 `nvidia-smi`、CUDA、Python、内存、磁盘挂载、网络与可用算力，克隆：

`git clone --branch codex/ouro-candidate-intervention-20261105 https://github.com/Gifoe/looped-transformer-stage1-reproduction.git`

将随附代码目录放入仓库的 `experiments/ouro_latent_state_mechanism_v1/`，执行 `README.md` 中的真实命令。固定模型 `ByteDance/Ouro-1.4B`，Revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`，原始 `model.safetensors` SHA256 `8872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`。先运行 `download_checkpoint.py` 下载核验，再运行 `prepare_arc.py`，以及 `verify_runtime.py` 完成输入和无干预 Hook 的技术校验。

保持历史 Ouro BF16、原始 SDPA、`use_cache=False`、无额外显式 Attention Mask、24 个物理层和四次共享权重循环；K1、K2、K3、K4 必须来自真实模型的 Native State，而不是由分数插值推断。Teacher Forcing 评分必须用完整词表 Log Softmax，按候选字母续写计算原始 Log Likelihood，不能换成自由生成或完整答案文本的字符归一化。

Gold 标签单独保存，模型推理程序不得读取 Gold。数据、示例、Prompt、Tokenizer 与模型来源都保存哈希。检查**6.1TB 实际资源**：如果是用户的数据卷，先用 `inventory_volume.py` 建立格式、容量和相关性清单；如果是存储容量，先运行 `storage_plan.py` 评估隐状态归档成本。ARC 数据本身不是 6.1TB，不能混淆“原始任务数据规模”和“中间激活存储”。

## Phase 1：立即以完整数据集建立循环状态图谱

不要先复现 B 的小样本，直接处理 ARC-Challenge 官方完整 Train、Validation、Test。先执行无干预原始排列 Joint-Option 的 K1–K4 采集；每题保存全部循环的四选项分数、预测、候选排序、正确性后验、Margin、Tie、Prompt Hash 与真实状态记录，按题分片落盘并支持断点恢复。

用随附的 `collect.py --capture current --save-head` 采集每个题目的当前 Question、全部 Options、标签位置和 `Answer:` 读出位置在全部 24 物理层、K1–K4 的后层隐状态，以及各循环结束后的归一化状态与完整词表输出。任何更细的 Attention/MLP/QKV 或完整 Few-Shot 上下文采集，应由 Astra 在检查容量与初轮现象后自行决定；不要先假定哪个 Token 或层最重要，也不要一次把 TB 级状态装入 GPU/内存。

随后运行 `analyze.py` 对**全量题目**进行配对转移分析：K1→K2、K2→K3、K3→K4 分别有多少救回、误伤、反转与边际收益？这些变化在不同题型和隐状态分布中如何出现？输出是否具有可重复的异常、非线性变化或难以由评分尺度解释的决策转移？

全量 Test 可用于已暴露 Benchmark 的描述性复现，但不要利用 Test Gold 反复择优并把结果称为新的盲测。新机制选择主要在全量 Train/Validation 上进行，训练示例的 25 道自包含题必须标识并从科学归因统计中排除。保留测试集的完整描述性报告。

**Phase 1 退出条件：** 新服务器真实复现 K1/K4 历史完整 Test 端点，补齐全量 K2/K3，生成逐层、逐循环、逐题行为和 latent-state 图谱。若端点不一致，优先审计输入、Tokenizer、评分器、并列规则、数值 Backend，不从不一致的运行中强行推断机制。

## Phase 2：不要预先指定机制，让数据生成问题

Astra 必须先读完整图谱，指出其中最能区分理论解释的真实观察，再提出假设。**不指定 H1/H2/H3/H4，也不规定必须解释为“语义比较”“标签绑定”或“早期表示不成熟”。** 方向性启发可以包括候选之间的信息交换、隐藏表示的重新组织、Readout 变化、分数尺度、循环内部的优化式动态，或上述范围外的新机制，但这些只是观察角度，不是事实或预注册真理。

每轮只选择足以提出区别性预测的少量假设；为每项列出：观察事实、竞争解释、会让自己改变判断的反例、最小可执行干预、阴性/等尺度对照以及验证划分。若某个变化只与救回相关、不能被干预或被改变尺度控制解释，就不得宣称因果机制。

Astra 需要对照现有 Ouro、Looped Transformer、Latent Reasoning 和 Mechanistic Interpretability 文献，防止重复已有 Hidden-State 可视化、Attention 图或跨循环移植结论。可自主检索最新文献，而非被本 Prompt 罗列的文献列表限制。

## Phase 3：Astra 指挥 Sol 的自主科研迭代循环

每轮执行如下状态机：

**完整观察 → Astra 提出科学问题 → 明确可否证预测 → Astra 下达 `orders/ROUND_N.md` → Sol 执行 → 数值/对照审计 → Astra 判断保留、改写或否定 → 下一轮。**

`orders/ROUND_N.md` 应指定相关的全量数据分层、精确 Loop/Layer/Token 选择、为什么选这些位置、实际 Hook 或 Probe、必要对照、失败条件、硬件和磁盘开销、预期能排除什么解释。Sol 必须交付真实运行结果和失败样本，不得根据结果私自翻转符号、选择层、改阈值或删题；Astra 对下一轮科研设计拥有最终权力。

当现有假设全部不成立时，**不许停止在“假设失败”的一句话上，也不许机械地随机搜索更多校正公式**。Astra 必须重新审查原观察本身是否稳定，然后主动改变观察尺度或科学问题：例如从最终分数转向层级残差、从单个答案位置转向候选间信息流、从相关性转向受控反事实、从某个单层转向跨层路径，或设计最小合成反例。必要时扩展到另一选择题数据集，仍使用同一冻结模型，并保持 Train/Validation/新独立验证划分。若多个解释始终无法区分，明确写出不可识别边界与下一项真正有辨识力的实验，不要伪造确定机制。

允许 Astra 自主调整采集粒度、增加原创分析代码、训练仅用于可读性测量的独立 Probe、设计针对隐藏状态的必要性与救援干预，并让 Sol 在服务器上执行。任何 Probe 必须防止训练和测试泄漏；所有 Patch 必须包含 Identity、Same-Shape、来源置换、范围或幅度匹配等恰当对照。单纯把 K2 的最终正确状态复制到 K1 末端不能作为已解释机制的证据。

**K3/K4 的用途由结果决定：** 主要关注 K1→K2 是否出现大幅跃迁；K2→K3、K3→K4 用于分析饱和、修复、回退或后续动态，不要求平均资源投入；如果全量数据显示 K1→K2 并非主区间，必须修改研究重心。

## Phase 4：交付和收敛

在新目录持续保存：`REPRODUCTION.md`、`FULL_COHORT_TRAJECTORIES.md`、`DISCOVERY_LOG.md`、`orders/ROUND_N.md`、`results/ROUND_N.md`、`CAUSAL_EVIDENCE.md` 和 `FINAL_RESEARCH_REPORT.md`，以及可重现源码、输入哈希、数值文件、必要矢量分片与反例图。小型代码和报告提交新 Git 分支；Checkpoint、TB 级激活和凭据不要推送到 GitHub。

研究成功的标准是出现有对照、可否证、可迁移的机制解释，或者得出有清晰证据支持的否定结论及边界。不要仅因实验次数多或 Accuracy 上升就认定成功。Astra 根据每轮结果决定下一步，直到进一步实验的信息价值不再足够，给出明确终止报告。

**立即行动：** Sol 从真实服务器检查、模型下载与 SHA 验证开始，运行全量 Train/Validation/Test 的 K1–K4 Joint-Option 和 Latent-State 采集，不复现 Experiment B 的 24 题 Benchmark；Astra 在完整首轮图谱形成后自主发现机制假设、连续下令并迭代。
