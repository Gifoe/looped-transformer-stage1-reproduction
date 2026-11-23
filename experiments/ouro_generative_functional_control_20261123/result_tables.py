"""Render training and matched evaluation tables from recorded run artifacts."""

import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value):
    return f"{100 * value:.2f}%"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    return "\n".join([*lines, *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def training(results, out):
    rows = []
    order = ("gen_only", "gen_rank", "multitask_small", "multitask_reasoning_small", "multitask_full",
             "retention_control", "retention_strong")
    for arm in order:
        path = results / arm / "TRAIN.json"
        if not path.exists():
            continue
        run = read(path)
        rows.append((arm, run["unique_samples"], run["steps"], f'{run["supervised_tokens"]:,}',
                     f'{run["padded_token_loops"]:,}', f'{run["seconds"]:.1f}',
                     f'{run["peak_cuda_allocated_bytes"] / 1e9:.3f}'))
    ledger = Path("GPU_LEDGER.tsv").read_text().splitlines()
    completed = [line.split("\t") for line in ledger if line.startswith("END\t")]
    seconds = sum(int(parts[4]) for parts in completed)
    planned_splits = [parts for parts in completed if parts[1].startswith("test_k4_core_") and parts[5] == "143"]
    attempted_failures = [parts[1] for parts in completed if parts[5] != "0" and parts not in planned_splits]
    body = ["# 训练记录", "",
            "单卡 RTX 4090 D，Ouro-1.4B Base 冻结。训练时循环使用 K2、K3、K4。",
            "轮次、实际配额和锁定样本见 `results/ROUND4_DECISION.json`、`data/` 及各训练目录。",
            "", table(("权重", "不同样本 ID", "更新步数", "目标 token", "骨干填充 token×loop", "训练秒", "峰值 GB"), rows),
            "", "完整池有 40,000 个独立题干或指令。No Robots 存在一对上游 ID 冲突，详见 `DATASET.md`。",
            "", f"GPU 作业时间合计 {seconds:,} 秒，包含模型加载、验证、确认及失败尝试。训练表只计训练函数记录的时间。",
            f"计划内拆分 {len(planned_splits)} 次，其他非零退出 {len(attempted_failures)} 次。逐次耗时、退出码见 `GPU_LEDGER.tsv`。", ""]
    (out / "TRAINING.md").write_text("\n".join(body), encoding="utf-8")


def confirmation(results, out):
    cells = read(results / "PAIRED_CONFIRMATION.json")
    seal = read(results / "FINAL_PROTOCOL.json")
    chosen = seal["selected_arm"]
    arms = seal["allowed_run_order"]
    dashboards = []
    columns = ("dataset", "depth", "mode", "arm", "metric", "reference", "n", "accuracy", "correct_count",
               "reference_accuracy", "delta", "improved", "harmed", "ci95_lower", "ci95_upper", "mcnemar_p",
               "seconds", "ms_per_item", "generated_tokens", "model_calls", "padded_token_loops", "cuda_peak")
    with (results / "CONFIRMATION.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for cell, data in cells.items():
            dataset, depth, mode = cell.split("__")
            for arm, result in data.items():
                cost = result["cost"]
                doc_count = result["summary"].get("valid", result["summary"].get("acc"))["n"]
                for comparison, name in (("native_paired", "native"), ("ranking_baseline_paired", "rank_only")):
                    for metric, value in (result[comparison] or {}).items():
                        writer.writerow((dataset, depth, mode, arm, metric, name, value["n"], value["controller"],
                                         result["summary"][metric]["count"], value["reference"], value["delta"],
                                         value["rescued"], value["harmed"], *value["paired_ci95"], value["exact_mcnemar"],
                                         cost["seconds"], 1e3 * cost["seconds"] / doc_count, cost.get("generated_tokens"),
                                         cost["model_calls"], cost["backbone_padded_token_loops"], cost["cuda_peak"]))
            summaries = {"letter": ("strict", "loose_correct", "valid"), "content": ("correct", "strict", "valid"),
                         "candidate": ("acc", "acc_norm"), "span": ("correct", "f1"), "gsm_final": ("correct", "strict"),
                         "gsm_process": ("correct", "strict", "process"), "other": ("valid", "completed", "rouge_l", "constraint")}
            for metric in summaries[mode]:
                if chosen not in data or metric not in data[chosen]["summary"]:
                    continue
                n = data[chosen]["summary"][metric]["n"]
                dashboards.append((dataset, depth, mode, metric, n, *(percent(data[arm]["summary"][metric]["value"])
                                     if arm in data and metric in data[arm]["summary"] else "—" for arm in arms)))
    body = ["# 确认集生成与候选评分", "", f"开发集锁定的唯一权重为 `{chosen}`。所有值均使用同一冻结协议。", "",
            "同题改对、改错、95% 配对区间、精确 McNemar、时长、峰值显存和计算量见 `results/CONFIRMATION.csv`。",
            "候选评分使用完整候选文本。生成从完整词表逐 token 贪婪解码。严格格式、宽松解析和内容得分分别记录。",
            "候选榜单与生成结果之间的差值仅在各自评分协议内解释。",
            "", table(("任务", "深度", "模式", "指标", "n", *arms), dashboards), ""]
    (out / "GENERATION_RESULTS.md").write_text("\n".join(body), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--training-only", action="store_true")
    a = parser.parse_args()
    training(a.results, a.out)
    if not a.training_only:
        confirmation(a.results, a.out)


if __name__ == "__main__":
    main()
