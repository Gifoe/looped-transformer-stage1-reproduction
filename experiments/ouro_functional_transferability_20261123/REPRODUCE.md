# 单卡复现

## 固定依赖

使用 Python 3.12.3、PyTorch 2.7.1+cu128、Transformers 4.55.4、NumPy 1.26.4、SciPy 1.15.3、PyArrow 21.0.0 和 Matplotlib 3.9.0。模型固定为 Ouro-1.4B Base 修订 `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`，模型权重 SHA-256 为 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`。用 BF16、SDPA、无 KV cache，固定候选 token 打分。 [官方模型快照](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1) · [实际环境、模型及 tokenizer 文件哈希](results/PREFLIGHT.json)

从父提交 `a07698897bfa705cf3d31deaa2d8c00050c5c84c` 取得原生 scorer 的依赖目录。下面变量为相应目录的绝对路径。模型快照应包含原生 `configuration_ouro.py`、`modeling_ouro.py` 和完整 tokenizer 文件，可与 preflight 核验。

| 变量 | 对应目录 |
| --- | --- |
| `archive` | `experiments/ouro_arc_pareto_20261119` |
| `previous` | `experiments/ouro_semantic_amplification_arc_20261122` |
| `prior_root` | `experiments/ouro_minimal_functional_feedback_20261123` |
| `latest` | `experiments/ouro_rank1_functional_theory_generation_20261123` |
| `model` | 固定的 Ouro-1.4B Base 快照目录 |
| `corpus` | ARC-Challenge 官方 parquet 目录，包含 `train.parquet`，用于历史四题数值一致性检查 |

在新工作目录执行下列命令。发布树中的 `data/` 保留选定问题和源清单。没有必要重取数据或引入答案以外的 QASC 支持字段。

```bash
sha256sum -c ARTIFACTS.sha256
export PYTHONPATH="$PWD:$latest:$prior_root:$previous:${PYTHONPATH:-}"
mkdir -p new_results
base=(--archive "$archive" --previous "$previous" --model "$model")
depth=("${base[@]}" --minimal "$prior_root" --out new_results/HOLDOUT)
t1="$PWD/reference_weights/t1.pt"
python preflight.py "${base[@]}" --corpus "$corpus" --rank1 "$t1" --out new_results/PREFLIGHT.json

for ds in arc csqa; do
  python functional_operator.py "${base[@]}" --train-data "data/${ds}_train.jsonl" \
    --n 96 --out "new_results/moments/$ds"
done
python depth_holdout.py "${depth[@]}" --phase train --source arc --suite depth --train-data data/arc_train.jsonl
python depth_holdout.py "${depth[@]}" --phase train --source arc --suite position --train-data data/arc_train.jsonl
python depth_holdout.py "${depth[@]}" --phase train --source csqa --suite source --train-data data/csqa_train.jsonl

python theory_prediction_round1.py "${base[@]}" --historical-rank1 "$t1" --data-root data \
  --results new_results --out new_results/forecast
forecast=(--forecast-lock new_results/forecast/FORECAST_LOCK.json --predictions new_results/forecast/PREDICTIONS.json)
python depth_holdout.py "${depth[@]}" --phase confirm --source arc --suite depth \
  --eval arc_confirm=data/arc_confirm.jsonl "${forecast[@]}"
python depth_holdout.py "${depth[@]}" --phase confirm --source arc --suite depth --groups k234 --eval-tag csqa_transfer \
  --eval csqa_confirm=data/csqa_confirm.jsonl "${forecast[@]}"
python depth_holdout.py "${depth[@]}" --phase confirm --source csqa --suite source \
  --eval arc_confirm=data/arc_confirm.jsonl --eval csqa_confirm=data/csqa_confirm.jsonl "${forecast[@]}"
python depth_holdout.py "${depth[@]}" --phase confirm --source arc --suite position \
  --eval arc_confirm=data/arc_confirm.jsonl "${forecast[@]}"
python state_horizon.py "${base[@]}" --rank1 "$t1" --data data/arc_confirm.jsonl --moment-root new_results/moments/arc \
  --forecast-lock new_results/forecast/FORECAST_LOCK.json --evaluation-name arc_confirm --out new_results/state_horizon/arc_confirm
python control_transport.py "${base[@]}" --minimal "$prior_root" --rank1 "$t1" --cohorts data/MANIFEST.json \
  --cohort arc_confirm --out new_results/D_TRANSPORT/arc_confirm --branch-controls
CUDA_VISIBLE_DEVICES="" python analyze_transfer.py --results new_results
```

第一轮确认使用与 checkpoint 匹配的预测 SHA。更换设备与重新训练可能重新生成 SHA，所以新复现先生成自己的锁，再开启确认。发布时将体积较大的原始功能矩 `.pt` 用 `gzip -n` 压缩。重用已发布矩时先解压，解压后的文件应与同目录 `.json` 的 `matrix_sha256` 相同。

## 第二轮

下列命令使用上文的 `base`、`depth` 和 `t1`。核验辅助脚本 [source_round2.sh](source_round2.sh) 记录了服务器加 GPU 台账的执行方式。

```bash
src() {
  case "$1" in
    arc|csqa) echo "data/$1_train.jsonl" ;;
    obqa|qasc) echo "data/optional/$1_train.jsonl" ;;
    mixed) echo data/arc_csqa_mixed_train.jsonl ;;
  esac
}
for ds in obqa qasc mixed; do
  python functional_operator.py "${base[@]}" --train-data "$(src "$ds")" --n 96 --out "new_results/moments/$ds"
done
for seed in 261123137 261123271; do
  for ds in arc csqa obqa qasc mixed; do
    if [[ "$seed" == 261123137 && ( "$ds" == arc || "$ds" == csqa ) ]]; then continue; fi
    entry=depth_holdout.py
    if [[ "$ds" == qasc ]]; then entry=depth_holdout_qasc.py; fi
    python "$entry" "${depth[@]}" --phase train --source "$ds" --suite source --train-data "$(src "$ds")" --seed "$seed"
  done
done
python theory_prediction.py "${base[@]}" --historical-rank1 "$t1" --data-root data \
  --results new_results --out new_results/forecast_round2 --round2
round2=(--forecast-lock new_results/forecast_round2/FORECAST_LOCK.json \
  --predictions new_results/forecast_round2/PREDICTIONS.json \
  --eval arc_round2_confirm=data/optional/arc_round2_confirm.jsonl \
  --eval csqa_round2_confirm=data/optional/csqa_round2_confirm.jsonl)
for seed in 261123137 261123271; do
  for ds in arc csqa obqa qasc mixed; do
    entry=depth_holdout.py
    suite=source
    if [[ "$ds" == qasc ]]; then entry=depth_holdout_qasc.py; fi
    if [[ "$ds" == arc && "$seed" == 261123137 ]]; then suite=depth; fi
    python "$entry" "${depth[@]}" --phase confirm --source "$ds" --suite "$suite" \
      --groups k234 --seed "$seed" --eval-tag r2 "${round2[@]}"
  done
done
```

使用确认前冻结的谱方向做有限剂量试验，以及使用 K4 完整词表贪心的短输出核查，复用 [gamma_queue.sh](gamma_queue.sh) 和 [generation_queue.sh](generation_queue.sh) 中的 Python 命令与相应锁文件。原始候选、生成及配对统计均随结果发布。

实跑在独立 GPU flock 下记账。整个新研究使用单张已授权的 RTX 4090 D，硬上限 28,800 秒，包括失败的测试。 [台账](GPU_LEDGER.tsv) · [发布审计](RELEASE_AUDIT.json)
