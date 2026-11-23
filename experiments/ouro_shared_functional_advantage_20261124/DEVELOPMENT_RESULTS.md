# 开发集结果

已核验 268／270 个条件的逐题回执。正式测试没有启动。

## 命中数

各格为同题同预算的命中数。`premature_eos` 统计提前停止次数，越少越好。SQuAD 原始开发集结果使用全部 64 题，其中 28 题段落与训练重合。模型选择表从同一组独立的 36 题重新计算。

| 任务 | K | 端点 | n | native | historical_rank | historical_mixed | ranking | generation | mixed | worst_advantage | anchored_policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc_challenge | 2 | candidate/acc_norm | 64 | 27 | 42 | 43 | 49 | 39 | 42 | 39 | 46 |
| arc_challenge | 3 | candidate/acc_norm | 64 | 35 | 44 | 43 | 49 | 41 | 42 | 43 | 47 |
| arc_challenge | 4 | candidate/acc_norm | 64 | 39 | 43 | 44 | 48 | 40 | 43 | 44 | 43 |
| csqa | 2 | candidate/acc_norm | 64 | 23 | 46 | 43 | 45 | 33 | 41 | 42 | 43 |
| csqa | 3 | candidate/acc_norm | 64 | 29 | 45 | 40 | 44 | 35 | 41 | 45 | 43 |
| csqa | 4 | candidate/acc_norm | 64 | 29 | 44 | 40 | 43 | 38 | 42 | 41 | 43 |
| arc_challenge | 2 | content/correct | 64 | 45 | 4 | 47 | 15 | 47 | 44 | 24 | 40 |
| arc_challenge | 3 | content/correct | 64 | 48 | 5 | 56 | 38 | 52 | 52 | 44 | 56 |
| arc_challenge | 4 | content/correct | 64 | 61 | 38 | 42 | 58 | 56 | 57 | 61 | 56 |
| csqa | 2 | content/correct | 64 | 29 | 4 | 40 | 20 | 37 | 38 | 36 | 35 |
| csqa | 3 | content/correct | 64 | 27 | 18 | 41 | 25 | 41 | 41 | 32 | 41 |
| csqa | 4 | content/correct | 64 | 30 | 22 | 41 | 35 | 44 | 46 | 37 | 43 |
| squad | 2 | span/correct | 64 | 16 | 31 | 44 | 39 | 42 | 38 | 41 | 42 |
| squad | 3 | span/correct | 64 | 29 | 35 | 49 | 37 | 44 | 42 | 42 | 44 |
| squad | 4 | span/correct | 64 | 30 | 34 | 50 | 39 | 42 | 41 | 41 | 44 |
| gsm8k | 2 | gsm_process/strict | 64 | 13 | 1 | 6 | 9 | 17 | 24 | 3 | 4 |
| gsm8k | 3 | gsm_process/strict | 64 | 21 | 13 | 20 | 26 | 36 | 37 | 5 | 待评测 |
| gsm8k | 4 | gsm_process/strict | 64 | 19 | 1 | 21 | 12 | 33 | 30 | 3 | 待评测 |
| gsm8k | 2 | gsm_process/process_correct | 64 | 13 | 1 | 3 | 9 | 16 | 24 | 3 | 1 |
| gsm8k | 3 | gsm_process/process_correct | 64 | 20 | 13 | 14 | 18 | 36 | 35 | 5 | 待评测 |
| gsm8k | 4 | gsm_process/process_correct | 64 | 14 | 0 | 14 | 3 | 33 | 26 | 3 | 待评测 |
| gsm8k | 2 | gsm_process/premature_eos | 64 | 1 | 1 | 33 | 13 | 26 | 15 | 57 | 43 |
| gsm8k | 3 | gsm_process/premature_eos | 64 | 0 | 18 | 12 | 6 | 11 | 14 | 55 | 待评测 |
| gsm8k | 4 | gsm_process/premature_eos | 64 | 1 | 1 | 12 | 1 | 10 | 15 | 36 | 待评测 |

## 第四轮与普通混合权重的配对差

区间为逐题配对 95% 区间，单位是百分点。提前 EOS 行的正差表示新增错误停止。GSM8K 的 K3/K4 尚无测量。

| 任务 | K | 端点 | n | 配对差及区间 pp | 新命中／丢失命中 |
| --- | --- | --- | --- | --- | --- |
| arc_challenge | 2 | candidate/acc_norm | 64 | +6.25 [0.00, 14.06] | 5/1 |
| arc_challenge | 3 | candidate/acc_norm | 64 | +7.81 [0.00, 15.62] | 6/1 |
| arc_challenge | 4 | candidate/acc_norm | 64 | +0.00 [-4.69, 4.69] | 1/1 |
| csqa | 2 | candidate/acc_norm | 64 | +3.12 [-7.81, 14.06] | 7/5 |
| csqa | 3 | candidate/acc_norm | 64 | +3.12 [-3.12, 9.38] | 3/1 |
| csqa | 4 | candidate/acc_norm | 64 | +1.56 [-3.12, 6.25] | 2/1 |
| arc_challenge | 2 | content/correct | 64 | -6.25 [-18.75, 6.25] | 7/11 |
| arc_challenge | 3 | content/correct | 64 | +6.25 [-3.12, 15.62] | 7/3 |
| arc_challenge | 4 | content/correct | 64 | -1.56 [-9.38, 6.25] | 3/4 |
| csqa | 2 | content/correct | 64 | -4.69 [-15.62, 6.25] | 5/8 |
| csqa | 3 | content/correct | 64 | +0.00 [-9.38, 9.38] | 4/4 |
| csqa | 4 | content/correct | 64 | -4.69 [-12.50, 1.56] | 1/4 |
| squad | 2 | span/correct | 64 | +6.25 [-3.12, 15.62] | 7/3 |
| squad | 3 | span/correct | 64 | +3.12 [-4.69, 10.94] | 4/2 |
| squad | 4 | span/correct | 64 | +4.69 [-3.12, 12.50] | 5/2 |
| gsm8k | 2 | gsm_process/strict | 64 | -31.25 [-43.75, -18.75] | 2/22 |
| gsm8k | 2 | gsm_process/process_correct | 64 | -35.94 [-48.44, -23.44] | 1/24 |
| gsm8k | 2 | gsm_process/premature_eos | 64 | +43.75 [26.56, 59.38] | 35/7 |

[完整统计与实际成本](results/PAIRED_DEVELOPMENT.csv) · [逐题原始结果](results/development.tar.gz) · [开发集训练消融](DEVELOPMENT_ABLATIONS.md)
