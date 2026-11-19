# Causal compositional dynamics in Ouro-1.4B

[中文报告](REPORT_ZH.md) · [冻结协议](PROTOCOL.md) · [推进判定](results/gate.json) · [六组图](results/figures)

本实验使用父提交 2afc5a51129fc506fbb982c3914477712a511e9b 中的修正版清单与原生钩子，在 Ouro-1.4B 的 K1 归一化输出上做来源 S、映射 M 的实体级四条件干预。使用一张 RTX 4090，模型参数冻结，BF16、SDPA、无 KV cache。

## 复现

1. 按[组织实验](../ouro_latent_organization_20261118/README.md)验证或生成修正版 groups.jsonl，SHA-256 为 4a255369c16f6fdf52d4f2569f47097a10044f4379ae533d8fc21587f96f608d。准备[扩展因果实验](../ouro_expanded_causal_20261118/README.md)的 inputs.jsonl、competence/inputs.jsonl、screened/pool.jsonl。它们用于语义签名去重。新世界的完整清单也随分支提供。
2. 使用固定 ByteDance/Ouro-1.4B Base 修订 574fa66cb8bf5abdc979642d01cf2b79b16bfab1。模型 SHA-256 为 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af。运行时会核对模型文件及配置、实现和分词资源的固定版本。
3. 在有单张 RTX 4090 或 5090 的 Linux 环境运行。对应依赖在 [环境记录](logs/dependencies.txt)，本次为 Python 3.12.3、Torch 2.7.1+cu128、Transformers 4.55.4。原始阵列需约 8 GB 磁盘，完整拟合权重和拟合中间文件另需约 1.5 GB。

```bash
export OURO_PYTHON=/absolute/path/to/venv/bin/python
bash experiments/ouro_causal_composition_20261119/reproduce.sh \
  /absolute/path/to/corrected-groups.jsonl \
  /absolute/path/to/fixed-model \
  /absolute/path/to/new-output-directory \
  /absolute/path/to/expanded-causal-output
```

脚本依次运行三类技术预检、独立生成、旧组采集、加密散列锁定模型与协议、独立采集、逐组评价、交互与对照分析、出图。复现目录必须是新目录。各任务并发采集时仍只使用同一张卡。

## 产物

| 路径 | 内容 |
|---|---|
| [data/fresh_groups.jsonl](data/fresh_groups.jsonl) | 480 个新组的原始提示、角色、语义值、答案和四世界/无关世界 |
| [development](development) 与 [independent](independent) | 经过分词的四世界、完整干预位置和剂量、K1–K4 分数、全宽原始阵列清单及训练前采集记录 |
| [frozen](frozen) | 冻结的参数网格、模型选择、权重 SHA-256、角色尺度、最终候选的可直接加载 NPZ |
| [evaluation](evaluation) 与 [results](results) | 自然、三种干预及交互的逐组预测，分区误差和组级区间，探针、答案变化、被批准的推进判断 |
| [results/cases/examples.json](results/cases/examples.json) | 六种选择规则独立列出的 18 个实例，含原题、干预元数据和实际/预测响应 |
| [logs](logs) | GPU、依赖、运行、审计日志 |

服务器输出根目录为 /root/autodl-tmp/ouro-causal-composition-20261119。实际 K1 全 token 状态以原生 BF16 位模式存放在各 cohort/task 的 token_k1.bf16，四世界按组连续，偏移量由 groups.jsonl 索引，有效长度为 tokens。等宽投影文件 token_k1.f16 的顺序相同。role_k2.npy 的轴为组、捕获条件、角色、宽度，条件顺序由 capture_composition.py 的 ARMS 固定。atlas.npy 的轴为组、对照条件、循环、物理层或归一化、角色、S/M/SM/I，未采集处为 NaN。函数 gather 捕获 K1 边界，全宽角色响应来自未改权重的原生 K2。完整数组和其余基线权重留在服务器，所有哈希都随分支发布。

真实 K4 分数直接来自 Ouro。所有代理语义指标和预测 K2 margin 使用真实 clean K2 锚点。预测 K4 margin 由同一 fit-only 诊断器产生，其输入为预测 K2 响应。冻结状态模型没有完整 token K2 解码头，因此没有做代理 K2 闭环替换。冻结主表保存原始 argmax 索引，答案支持用连续 margin 计算。报告和图 5 的翻转统计仅在 clean 与 edited 均有唯一最高分时计入，分母是整个组集合，并列频数单独发布。
