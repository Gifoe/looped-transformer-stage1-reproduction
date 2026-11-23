"""Create the compact Chinese report tables from the sealed test results."""

from pathlib import Path

from result_tables import percent, read, table


METRICS = {
    "content": ("correct", "完整选项内容"),
    "letter": ("strict", "严格字母"),
    "candidate": ("acc_norm", "候选 acc_norm"),
    "span": ("correct", "生成 EM"),
    "gsm_final": ("correct", "数字正确率"),
    "gsm_process": ("strict", "过程生成的严格数字正确率"),
    "other": ("rouge_l", "ROUGE-L"),
}


def signed(value):
    return f"{100 * value:+.2f} pp"


def contrast(data):
    low, high = data["paired_ci95"]
    return f"{signed(data['delta'])} [{100 * low:.2f}, {100 * high:.2f}]"


def main():
    root = Path("results")
    protocol = read(root / "FINAL_PROTOCOL.json")
    paired = read(root / "PAIRED_CONFIRMATION.json")
    transfer = read(root / "MATCHED_DEPTH_TRANSFER.json")
    chosen = protocol["selected_arm"]
    rows = []
    for cell, arms in paired.items():
        dataset, depth, mode = cell.split("__")
        if depth != "k4" or chosen not in arms or "rank_only" not in arms:
            continue
        name, display = METRICS[mode]
        own = arms[chosen]
        native = own["native_paired"][name]
        old = own["ranking_baseline_paired"][name]
        gain = f"{native['rescued']}/{native['harmed']}"
        before = 1000 * arms["native"]["cost"]["seconds"] / native["n"]
        after = 1000 * own["cost"]["seconds"] / native["n"]
        rows.append((dataset, display, native["n"], percent(native["reference"]), percent(old["reference"]),
                     percent(native["controller"]), contrast(native), gain, contrast(old), f"{before:.1f} / {after:.1f}"))
    verification = []
    for cohort, cell in transfer.items():
        ds, mode = cohort.split("/")
        if mode == "other":
            continue
        verified = [cell["depths"][f"k{k}"][chosen]["native"]["delta"] for k in range(2, 7)]
        verification.append((ds, METRICS[mode][1], len(cell["ids"]), *[signed(x) for x in verified]))
    dev = protocol["development_evidence"]
    selectors = [(a, percent(data["semantic_macro"]), percent(data["strict_letter"]),
                  percent(data["valid_letter"]), percent(data["paired_candidate_macro"])) for a, data in dev.items()]
    counts = read(root / "K4_FAILURE_AUDIT.json")
    failure_rows = []
    for key, data in counts.items():
        ds, arm = key.split("/")
        if arm not in {chosen, "rank_only", "native"}:
            continue
        c = data["counts"]
        failure_rows.append((ds, arm, c.get("scores_correct_content_wrong", 0), c.get("generation_correct_candidate_wrong", 0),
                             c.get("letter_lowercase", 0), c.get("letter_only_loose", 0), c.get("content_label_conflict", 0),
                             c.get("content_hit_limit", 0)))
    content = ["## 锁定权重", "", f"开发集选择 `{chosen}`。检查点 SHA-256 为 `{protocol['checkpoint_sha256']}`。",
               "", table(("控制器", "开发生成语义宏平均", "严格字母宏平均", "字母有效率宏平均", "候选 acc_norm 宏平均"), selectors),
               "", "## K4 确认", "",
               "差值后的方括号给出 95% 逐题配对区间。正负计数以 Native 为参照。ROUGE-L 的正负计数指分数升降。延迟为批量总时长除以题数。",
               "", table(("任务", "指标", "n", "Native", "历史排序", "锁定权重", "对 Native 差值及区间", "改好/改坏",
                          "对历史排序差值及区间", "Native / 锁定 ms/题"), rows),
               "", "## 同题深度外推", "", "所有深度使用同一组 ID，K4 同题子集只是用来分析深度。每格是锁定权重相对 Native 的配对差。",
               "", table(("任务", "指标", "n", "K2", "K3", "K4", "K5", "K6"), verification),
               "", "## 候选与生成分歧", "",
               table(("任务", "方法", "候选对内容错", "候选错内容对", "小写首选项", "宽松对且严格错", "内容与字母冲突", "内容到长度上限"), failure_rows),
               "", "完整预测、严格和宽松格式、区间、McNemar、模型调用及显存见 `GENERATION_RESULTS.md` 和 `results/CONFIRMATION.csv`。",
               "同题深度的区间、逐题失败示例见 `results/MATCHED_DEPTH_TRANSFER.json` 和 `results/K4_FAILURE_AUDIT.json`。", ""]
    target = Path("REPORT_ZH.md")
    intro = target.read_text(encoding="utf-8").split("\n## 锁定权重\n")[0] if target.exists() else "# Ouro-1.4B 生成控制重训"
    target.write_text(intro.rstrip() + "\n\n" + "\n".join(content), encoding="utf-8")


if __name__ == "__main__":
    main()
