# 单卡复现

使用 Linux、Python 3.12.3 和支持 BF16 的单张 CUDA GPU。实测环境为 RTX 4090 D、Torch 2.7.1+cu128、Transformers 4.55.4、NumPy 1.26.4、SciPy 1.15.3、PyArrow 21.0.0、tokenizers 0.21.4、safetensors 0.8.0。历史评分器来自本分支同一 `experiments/` 下的归档。克隆完整分支，保留相对目录结构。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.55.4 numpy==1.26.4 scipy==1.15.3 pyarrow==21.0.0 tokenizers==0.21.4 safetensors==0.8.0
hf download ByteDance/Ouro-1.4B --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 --local-dir /data/ouro-base
```

运行以下命令会复用发布权重并重新计算两套完整候选评分、自由生成、梯度及因果分析。另一次运行把 `frozen` 换成 `train`，用新的空输出目录从 ARC Train 开始训练和选型。

```bash
cd experiments/ouro_minimal_functional_feedback_20261123
sha256sum -c --quiet ARTIFACTS.sha256
CUDA_VISIBLE_DEVICES=0 bash reproduce.sh frozen /data/ouro-base /data/arc-parquet /scratch/ouro-minimal-frozen
CUDA_VISIBLE_DEVICES=0 bash reproduce.sh train /data/ouro-base /data/arc-parquet /scratch/ouro-minimal-train
```

脚本下载固定版本的 ARC Parquet 缺失文件并校验三个 split 的 SHA-256。可把官方 OpenBookQA Test 原始 Parquet 作为末尾可选参数，便于受限网络运行。脚本校验 OpenBookQA 的规范化文件，旧方法的候选分数直接从归档解压。旧方法生成使用原权重，重新计算以获得与当前模型严格相同的全量题面、解析和贪心程序。Ouro 加载器校验模型文件和分词器哈希，不需要联网加载模型。

`results/FREEZE.json` 列出锁定的共享模块、剂量、源代码及权重指纹。发布了所有训练权重，包括 rank、目标和容量对照。候选逐题输出在 `results/RESULTS_ARC.jsonl` 和 `results/RESULTS_OPENBOOKQA.jsonl`。自由生成汇总与计时在 `results/generation/`，归档的完整逐题自由生成在 `results/GENERATION_RESULTS.jsonl`。复现过程中生成文件写在输出目录的 `results/generation/`。训练、评测和 GPU 机制阶段串行，`GPU_LEDGER.tsv` 记录保守的五小时阶段墙钟上限。计时依赖 GPU、软件和批次。
