# T1 单卡复现

使用历史 4,096 参数 rank1_t1 权重。权重与原实验冻结记录一致。候选任务与前一轮 Mixed 完全相同，MMLU-Pro 直接复用并重新解析 197 批原始生成。

在本实验根目录配置前一轮 REPRODUCE.md 的环境与归档。保留 Mixed 的 FREEZE.json 和逐题文件。将 T1 评测代码及权重复制到一个干净运行目录的 t1_full 子目录。七项候选评分支持使用同一冻结回执和逐分片回执断点续跑。

    source external_env.sh
    "$py" t1_full/prepare_archive.py --stage2 "$feedback/stage2"
    (cd t1_full && sha256sum -c WEIGHTS.sha256)
    export OURO_T1_WEIGHT="$PWD/t1_full/weights/rank1_t1.pt"
    mkdir -p t1_full/results t1_full/logs
    bash t1_full/run_t1.sh
    for p in mixed_full/results/predictions/*.jsonl.gz; do
      test -e "${p%.gz}" || gzip -dk "$p"
    done
    "$py" -u t1_full/aggregate_t1.py --final --out t1_full/results --mixed mixed_full/results \
      --model "$model" --minimal "$feedback" --stage2 mixed_full/archive --unified "$unified" \
      --followup "$feedback/generation_followup"

原始 T1 权重来源与参数信息见历史 [冻结回执](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/research/ouro-functional-controllability/experiments/ouro_functional_controllability_20261123/results/FREEZE.json)。
