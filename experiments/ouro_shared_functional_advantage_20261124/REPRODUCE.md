# 复现

使用单张 4090 D、Python 3.12、PyTorch 2.7.1+cu128、Transformers 4.55.4、BF16、SDPA、无 KV cache。原生评分器验证 Ouro 权重、模型代码和 tokenizer 的 SHA。父研究脚本和冻结权重来自同一仓库。 [父研究配置](../ouro_generative_functional_control_20261123/REPRODUCE.md)

在新的工作目录设置 ADVANCE_GENERATION_ROOT 为父生成实验目录，将 GEN_TRANSFER_ROOT 指向父 transfer 目录，将 dynamics 指向父 transport 目录。按父研究复现说明取得四个固定官方训练输入和已有冻结权重。每个新划分的入选 ID、源 SHA、前作训练暴露计数和长度上限见 data/DATA_LOCK.json。

    source control_env.sh
    mkdir -p data results
    sha256sum PROTOCOL.md > PROTOCOL.sha256
    CUDA_VISIBLE_DEVICES='' "$py" prepare_data.py --model "$model" --arc "$arc_train" --csqa "$csqa_train" --squad "$squad_train" --gsm8k "$gsm8k_train" --prior "$gen/data" --history "$prior_transfer/data" --history "$dynamics/data" --out data
    CUDA_VISIBLE_DEVICES='' "$py" trajectory_audit.py --archive "$gen/results/confirmation" --out results/TRAJECTORY_AUDIT.json
    common=(--archive "$archive" --previous "$previous" --model "$model" --data data)
    bash control_budget.sh moment_fit collect_moments.py "${common[@]}" --out results/MOMENT_FIT --fold moment_fit > moment_fit.log 2>&1
    bash control_budget.sh moment_holdout collect_moments.py "${common[@]}" --out results/MOMENT_HOLDOUT --fold moment_holdout --depths 2 3 4 5 6 > moment_holdout.log 2>&1
    CUDA_VISIBLE_DEVICES='' "$py" shared_direction.py --fit results/MOMENT_FIT --out results/FIT_SHARED
    mkdir -p results/MOMENT_CANDIDATE_FIT
    for f in "$PWD"/results/MOMENT_FIT/*__candidate__*.pt; do ln -s "$f" results/MOMENT_CANDIDATE_FIT/; done
    CUDA_VISIBLE_DEVICES='' "$py" shared_direction.py --fit results/MOMENT_CANDIDATE_FIT --out results/FIT_CANDIDATE
    bash run_mechanism.sh > run_mechanism.log 2>&1
    bash run_fitting.sh > run_fitting.log 2>&1
    CUDA_VISIBLE_DEVICES='' "$py" make_weak_axis.py --axis results/FIT_SHARED/DIRECTIONS.pt --reference "$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt" --out results/fits/linear_weak/controller.pt
    bash run_development.sh > run_development.log 2>&1

正式确认从父研究的四个官方评测样本文件重建，用 overlap_audit.py 检查新训练池及两个历史训练池。新确认开始前，使用 freeze_eval.py 锁定唯一部署权重、输入指纹、评分代码和每项样本量。旧重训评测的暂停状态与本实验独立。

    mkdir -p data/official
    for ds in arc_challenge csqa squad gsm8k; do ln -s "$gen/data/test_$ds.jsonl" data/official/; done
    CUDA_VISIBLE_DEVICES='' "$py" "$gen/overlap_audit.py" --data data/official --train data/arc_challenge_fit.jsonl --train data/csqa_fit.jsonl --train data/squad_fit.jsonl --train data/gsm8k_fit.jsonl --train "$gen/data/csqa_matched.jsonl" --train "$gen/data/full_mixed_train.jsonl" --out data/confirmation
    CUDA_VISIBLE_DEVICES='' "$py" lock_passage_split.py --current_fit data/squad_fit.jsonl --historical_fit "$gen/data/full_mixed_train.jsonl" --development data/squad_development.jsonl --confirmation data/confirmation/test_squad.jsonl --out results/DEVELOPMENT_EXCLUSIONS.json
    bash run_round3.sh > run_round3.log 2>&1
    bash run_anchored.sh > run_anchored.log 2>&1
    bash seal_confirmation.sh
    bash run_confirmation.sh > run_confirmation.log 2>&1
    bash run_postprocess.sh

发布包保存源文件哈希、矩摘要、可逐题重算的结果和实际训练权重。完整矩阵可按前述命令重建。逐题结果解压后，应与各条件回执 SHA 一致。训练、机制前向、验证与测试的 GPU 耗时均计入 GPU_LEDGER.tsv。

从已保存的矩阵、权重和开发结果可独立重算训练后的一阶斜率，无需运行模型。参数中的历史权重路径应与开发回执的 SHA 相符。

    CUDA_VISIBLE_DEVICES='' "$py" audit_deployed_directions.py --fit results/MOMENT_FIT --holdout results/MOMENT_HOLDOUT --weights results/fits --paired results/PAIRED_DEVELOPMENT.json --development results/development --historical "historical_rank=$prior_transfer/results/HOLDOUT/csqa/source/s261123137/k234/controller.pt" --historical "historical_retention=$gen/results/FINAL_CONTROLLER.pt" --historical "historical_mixed=$gen/results/multitask_full/controller.pt" --out results/DEPLOYED_DIRECTIONS.json
