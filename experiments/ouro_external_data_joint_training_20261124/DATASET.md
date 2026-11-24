# 训练与评测数据

| 监督来源 | 官方 Train | 清洗后合格 | 拟合题数 | 保留开发题数 | 训练视图 |
| --- | ---: | ---: | ---: | ---: | --- |
| CSQA | 9,741 | 8,558 | 1,000 | 360 | 完整选项文本约 70%，字母约 30% |
| OBQA-main | 4,957 | 4,775 | 1,000 | 360 | 完整选项文本约 70%，字母约 30% |
| GSM8K-main | 7,473 | 7,328 | 2,000 | 360 | 完整过程 75%，最终答案 25% |
| ScienceQA | 12,726 | 2,192 | 1,000 | 360 | C 短答，D 短答及原始解释各 50% |

官方原始文件修订和 SHA、实际题号、每种视图的目标 token 数以及排除名单保存在 `data/MANIFEST.json` 和 `data/EXCLUSIONS.jsonl`。每组共 4,000 个独立原题。CSQA 与 OBQA 的短答格式沿用历史代码的固定哈希选型。GSM8K 每批四个原题中，三个使用完整过程，一个使用最终答案。按过程长度四分位均匀抽取。原始答案及删去计算器标记的训练目标均保存，截断过程无法进入两个视图的共同拟合池。 [CSQA 官方修订](https://huggingface.co/datasets/tau/commonsense_qa/tree/d90214517943631510b39b7d2503b7ea93ec09c2) · [OBQA 官方修订](https://huggingface.co/datasets/allenai/openbookqa/tree/8c4888010de9b23a2d136ebd31d5f5ea9f3d90fa) · [GSM8K 官方修订](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866)

ScienceQA 用 GitHub 修订 `2cbf8318e07b9ece895bb2ae605e71e38d623264` 的 `problems.json` 和 `pid_splits.json`。排除 6,717 道图片或视觉指代题、1,243 道无法由原解释中的答案文本或明确答案标签支持的题、658 道无答案解释或无有效标签的题、327 道有明显重合选项或集合式选项的题。另按共同规则去重。选项保留原顺序以维持原解释中的字母与方位指代。文本规则能验证表面答案一致性，无法保证所有解释逻辑正确。 [ScienceQA 官方修订](https://github.com/lupantech/ScienceQA/tree/2cbf8318e07b9ece895bb2ae605e71e38d623264/data/scienceqa)

训练池跨来源合并规范化题干相同或词三元组 Jaccard 至少 0.8 的组件。带有任一已知评测题或历史保留题匹配的组件整组排除。组件是在原题层级形成的，随后同一题的全部训练视图绑定到唯一划分。词三元组对不足六词的题只使用规范化精确匹配。改写造成的潜在语义重叠和基座预训练历史仍不可核实。

开发筛选取预先排序的前 120 题。扩样取 ARC 官方 Validation 299 题以及其余各域固定 360 题。SQuAD 开发来自 Train，每段落只抽一题，排除可查历史实际训练、既有开发、矩阵留出和完整官方 Validation 的段落。公开历史测试重跑覆盖原七项 31,737 题、MMLU-Pro 12,032、CSQA 作者 Validation 1,221、OBQA Test 500、SQuAD Validation 10,570、GSM8K Test 1,319。CSQA、OBQA、GSM8K 的陈述限于各自未见题迁移。ARC、SQuAD 与其余未进入新拟合池的数据源用于检验跨来源迁移。T1/T2 的 ARC Train 谱系单独标注。
