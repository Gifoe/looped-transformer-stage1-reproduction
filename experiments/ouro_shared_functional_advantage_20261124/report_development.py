"""Render the completed development evidence without sealing an official test."""

import json
from pathlib import Path

from freeze_eval import ADVANTAGE, BASELINES, ENDPOINTS, select
from render_results import difference, read, table, training_tables


COMPLETED_PROPOSALS = (*ADVANTAGE, "teacher_resume", "on_policy_resume")
ARMS = (*BASELINES, *COMPLETED_PROPOSALS, "anchored_policy")
METRICS = (*ENDPOINTS, ("gsm8k", "gsm_process", "premature_eos"))


def main():
    root = Path("results")
    paired = read(root / "PAIRED_DEVELOPMENT.json")
    groups = paired["results"]
    missing = set()
    for key, cells in groups.items():
        if set(cells) - set(ARMS):
            raise ValueError(f"Unexpected development arms at {key}")
        missing.update(f"{arm}/{key}" for arm in set(ARMS) - set(cells))
    expected_groups = {f"{ds}/k{k}/{mode}" for ds, mode, _ in ENDPOINTS for k in (2, 3, 4)}
    if set(groups) != expected_groups:
        raise ValueError("An entire development condition is missing")
    if missing != {f"anchored_policy/gsm8k/k{k}/gsm_process" for k in (3, 4)}:
        raise ValueError(f"Unexpected pause state: {sorted(missing)}")
    if (root / "CONFIRMATION_LOCK.json").exists():
        raise ValueError("The official study has been sealed")
    _, validation, _ = select(root / "development", BASELINES, COMPLETED_PROPOSALS)
    training_tables(root, {"validation": validation}, groups, Path("DEVELOPMENT_ABLATIONS.md"),
                    "此表仅列已完成全部端点的比较组。第四轮 anchored_policy 缺少两项数学验证，未参加比较。")
    counts = []
    anchor = []
    columns = ("native", "historical_rank", "historical_mixed", "ranking", "generation", "mixed",
               "worst_advantage", "anchored_policy")
    for ds, mode, metric in METRICS:
        for k in (2, 3, 4):
            group = groups[f"{ds}/k{k}/{mode}"]
            counts.append((ds, k, f"{mode}/{metric}", group["native"]["metrics"][metric]["n"],
                           *(group[arm]["metrics"][metric]["count"] if arm in group else "待评测" for arm in columns)))
            if "anchored_policy" in group:
                detail = group["anchored_policy"]["paired"]["mixed"][metric]
                anchor.append((ds, k, f"{mode}/{metric}", detail["n"], difference(detail),
                               f"{detail['rescued']}/{detail['harmed']}"))
    coverage = {"status": "evaluation_paused", "development_receipts": paired["coverage"]["cells"],
                "development_expected": 18 * len(ARMS), "missing_development": sorted(missing),
                "official_evaluation_started": False, "deployment_checkpoint_selected": False}
    (root / "SNAPSHOT_STATUS.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    doc = ["# 开发集结果", "", f"已核验 {coverage['development_receipts']}／{coverage['development_expected']} 个条件的逐题回执。正式测试没有启动。", "",
           "## 命中数", "", "各格为同题同预算的命中数。`premature_eos` 统计提前停止次数，越少越好。SQuAD 原始开发集结果使用全部 64 题，其中 28 题段落与训练重合。模型选择表从同一组独立的 36 题重新计算。", "",
           table(("任务", "K", "端点", "n", *columns), counts), "", "## 第四轮与普通混合权重的配对差", "",
           "区间为逐题配对 95% 区间，单位是百分点。提前 EOS 行的正差表示新增错误停止。GSM8K 的 K3/K4 尚无测量。", "",
           table(("任务", "K", "端点", "n", "配对差及区间 pp", "新命中／丢失命中"), anchor), "",
           "[完整统计与实际成本](results/PAIRED_DEVELOPMENT.csv) · [逐题原始结果](results/development.tar.gz) · [开发集训练消融](DEVELOPMENT_ABLATIONS.md)", ""]
    Path("DEVELOPMENT_RESULTS.md").write_text("\n".join(doc), encoding="utf-8")


if __name__ == "__main__":
    main()
