# T2 单卡复现

固定历史 4,096 参数 rank1_t2，权重形式为 Rank-1、RMS 相对限幅 0.45，循环深度为 K4。七项零样本候选评分使用与 Mixed 和 T1 相同的完整请求。MMLU-Pro 直接复用并重新解析原始 T2 的 197 批真实贪心生成。

配置根目录复现文档中的 CUDA 与归档环境。保留 Mixed 和 T1 的冻结回执、代码和逐题文件。把 T2 评测代码及权重放到新的运行目录的 t2_full 子目录。正在运行的任务可在同一目录用分片回执断点续跑。

    source external_env.sh
    "$py" t1_full/prepare_archive.py --stage2 "$feedback/stage2"
    (cd t2_full && sha256sum -c WEIGHTS.sha256)
    export OURO_T2_WEIGHT="$PWD/t2_full/weights/rank1_t2.pt"
    mkdir -p t2_full/results t2_full/logs
    bash t2_full/run_t2.sh
    for p in {mixed_full,t1_full}/results/predictions/*.jsonl.gz; do
      test -e "${p%.gz}" || gzip -dk "$p"
    done
    "$py" -u t2_full/aggregate_t2.py --final --out t2_full/results --mixed mixed_full/results --t1 t1_full/results --model "$model" --minimal "$feedback" --stage2 mixed_full/archive --unified "$unified" --followup "$feedback/generation_followup"

历史 T2 生成谱系和原始结果见 [Rank-1 生成研究](https://github.com/Gifoe/looped-transformer-stage1-reproduction/blob/research/ouro-rank1-functional-theory-generation/experiments/ouro_rank1_functional_theory_generation_20261123/REPORT_ZH.md)。
