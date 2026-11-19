"""Generate the released benchmark tables, figures and artifact checksums."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_arc import load


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure(metrics, dest):
    styles = {
        "native": ("Native", "#34495e", "o"),
        "loopcd_fixed": ("LoopCD logits, fixed", "#ce733b", "s"),
        "loopcd_adaptive": ("LoopCD logits, adaptive", "#764394", "X"),
        "loopcd_hidden": ("LoopCD hidden", "#167d77", "^"),
        "secant": ("Clipped secant", "#b33962", "D"),
        "laststep": ("One-step extrapolation", "#8d734e", "v"),
    }
    fig, (axs, lat) = plt.subplots(1, 2, figsize=(10.3, 4.2), sharey=True, layout="constrained")
    for tag, (label, color, marker) in styles.items():
        entries = [(name, row) for name, row in metrics.items() if name.rsplit("_", 1)[0] == tag]
        y = [row["accuracy_pct"] for _, row in entries]
        axs.plot([row["linear_flop_proxy"] / 1e15 for _, row in entries], y,
                 marker=marker, color=color, label=label, lw=1.45, ms=5)
        lat.plot([row["cost"]["prefill_plus_decode_p50_ms"] for _, row in entries], y,
                 marker=marker, color=color, lw=1.45, ms=5)
    axs.set(xlabel="Linear layers + output projection (PFLOP proxy)", ylabel="ARC-Challenge acc_norm (%)")
    lat.set(xlabel="Cache-enabled prefill + one decode step, p50 (ms)")
    for ax in (axs, lat):
        ax.grid(alpha=.21)
        ax.spines[["top", "right"]].set_visible(False)
    axs.legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Ouro-1.4B Base on ARC-Challenge test · zero-shot · 1,172 questions")
    fig.savefig(dest / "accuracy_compute.png", dpi=220)
    fig.savefig(dest / "accuracy_compute.pdf")
    plt.close(fig)


def validate(out):
    config = read(out / "freeze.json")
    for name, digest in config["config_sha256"].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
    cohorts = {key: load(out / f"{key}_predictions.jsonl")
               for key in ("validation", "test", "oracle", "oracle_followup")}
    ids = {k: {row["id"] for row in v} for k, v in cohorts.items()}
    for key, n in (("test", 1172), ("validation", 299), ("oracle", 128), ("oracle_followup", 171)):
        assert len(cohorts[key]) == len(ids[key]) == n
        first = set(cohorts[key][0]["scores"])
        for row in cohorts[key]:
            assert set(row["scores"]) == first
            assert all(len(v) == len(row["labels"]) for v in row["scores"].values())
    assert not ids["test"] & ids["validation"]
    assert not ids["oracle"] & ids["oracle_followup"]
    assert ids["oracle"] | ids["oracle_followup"] == ids["validation"]
    runtime = read(out / "runtime.json")
    for key, expected in [("validation", 1194), ("test", 4687)]:
        record = read(out / f"{key}_execution.json")
        assert record["candidates"] == expected == sum(len(row["labels"]) for row in cohorts[key])
    assert runtime["model_config_layers"] == 24
    standalone = read(out / "standalone_cost.json")
    assert all(r["candidates"] == 4687 and r["max_abs_score_diff_vs_shared_run"] == 0 for r in standalone.values())
    trace = read(out / "latency.json")
    durations = Counter(x["method"] for x in trace["observations"])
    assert len(durations) == 17 and set(durations.values()) == {24}
    assert all(v["top1_equal"] for v in trace["parity"].values())
    return config


def compare(a):
    lo, hi = a["delta_bootstrap_95_pp"]
    return (f'{a["delta_pp"]:+.2f} 个百分点，95% 配对区间 [{lo:+.2f}, {hi:+.2f}]，'
            f'纠正 {a["wrong_to_correct"]} 题、退化 {a["correct_to_wrong"]} 题')


def release(out, dest):
    config = validate(out)
    verdicts = out / "test_verdicts.csv"
    verdicts.write_bytes(verdicts.read_bytes().replace(b"\r\n", b"\n"))
    data = read(out / "test_metrics.json")
    norm, raw = data["acc_norm"], data["acc"]
    oracle = read(out / "oracle_metrics.json")["acc_norm"]
    follow = read(out / "oracle_followup_metrics.json")
    hold = follow["metrics"]["acc_norm"]
    replay = read(out / "adaptive_depth_replay.json")["acc_norm"]
    standalone = read(out / "standalone_cost.json")
    test_run = read(out / "test_execution.json")
    best = max(norm, key=lambda method: norm[method]["correct"])
    best_raw = max(raw, key=lambda method: raw[method]["correct"])
    shallow = "loopcd_fixed_k3"
    assert not any(x["passes_screen"] for x in config["secant_screen"].values())
    assert not any(x["passes_screen"] for x in config["oracle_training_screen"].values())
    assert not any(x["passes_screen"] for x in follow["same_pilot_gate"].values())
    base = [
        "# Ouro-1.4B Base benchmarks",
        "",
        "Official ARC-Challenge test, 1,172 items and 4,687 complete continuations. Zero-shot, BF16, SDPA, RTX 4090. Validation selected the same-budget references before the test pass.",
        "",
        "| Method | Loops | acc | acc_norm | Prefill p50 ms | Decode p50 ms | Peak MiB | Full split s | Δ acc_norm vs native at K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in norm.items():
        cost = row["cost"]
        elapsed = f'{standalone[name]["wall_s"]:.2f}' if name in standalone else "—"
        base.append(f'| {name} | {row["loop_budget"]} | {raw[name]["accuracy_pct"]:.2f} | '
                    f'{row["accuracy_pct"]:.2f} | {cost["prefill_p50_ms"]:.2f} | '
                    f'{cost["decode_p50_ms"]:.2f} | {cost["peak_allocated_bytes"] / 2**20:.0f} | '
                    f'{elapsed} | {row["vs_same_depth_native"]["delta_pp"]:+.2f} pp |')
    base.extend([
        "",
        "Cached timings are synchronized wall-clock measurements of 24 fixed validation prompts at batch size 1. The cache grows at each recurrent layer. Full-split timing uses batch size 16 and teacher-forced answer candidates with no KV cache. A dash means the configuration has cached latency measurements and was scored in the shared sweep, with no standalone full-split timing. Linear FLOP proxies omit attention products, normalization, activations, padding and cache movement.",
        "",
        "![Accuracy and compute](accuracy_compute.png)",
    ])
    write(dest / "BASELINES.md", base)
    write(dest / "REPORT_ZH.md", [
        "# Ouro-1.4B 在完整 ARC-Challenge 上的结果",
        "",
        f'**测试集最高观测 acc_norm 为 {norm[best]["accuracy_pct"]:.2f}%（{norm[best]["correct"]}/1,172，{best}）。** '
        f'相对同深度原生 K4 为 {compare(norm[best]["vs_same_depth_native"])}。'
        f'最高原始 acc 为 {raw[best_raw]["accuracy_pct"]:.2f}%（{raw[best_raw]["correct"]}/1,172，{best_raw}）。'
        "测试集最高配置是描述性结果，验证集事先选定的 K4 基线为原生 K4。新测试的轨迹外推和跨轮移植都未通过开发门槛。",
        "",
        f'三轮固定 Logits LoopCD 的 acc_norm 为 {norm[shallow]["accuracy_pct"]:.2f}%，原生四轮为 {norm["native_k4"]["accuracy_pct"]:.2f}%。'
        f'差异为 {compare(norm[shallow]["vs_native_k4"])}。'
        f'其线性算量代理为原生四轮的 {100*norm[shallow]["linear_flop_proxy"]/norm["native_k4"]["linear_flop_proxy"]:.1f}%，'
        f'批量 1 的 prefill 加一次 decode 的实测中位数为 {norm[shallow]["cost"]["prefill_plus_decode_p50_ms"]:.2f} ms，'
        f'原生四轮为 {norm["native_k4"]["cost"]["prefill_plus_decode_p50_ms"]:.2f} ms。'
        "配对区间跨零，并且相同三轮预算下已有 LoopCD 与外推持平或领先，因此证据不足以确认新方法改善 Pareto 前沿。",
        "",
        "## 增加轮数有多大作用",
        "",
        *[f'- 原生 K{k} 为 {norm[f"native_k{k}"]["accuracy_pct"]:.2f}% acc_norm、{raw[f"native_k{k}"]["accuracy_pct"]:.2f}% acc。'
          + (f' 相对上一轮 {compare(norm[f"native_k{k}"]["vs_previous_native"])}。' if k > 1 else "")
          for k in range(1, 5)],
        "",
        f'验证集选出的置信度路由在测试集回放中得到 {replay["accuracy_pct"]:.2f}% acc_norm，平均预算 {replay["mean_loop_budget"]:.2f} 轮。'
        "这项回放的两套分数在完整轨迹中已采集，真实提前退出耗时尚未测量。",
        "",
        "## 跨轮干预与训练决策",
        "",
        "| 分量 | 128 题初筛 Δ vs 原生 K2 | 171 题复核 Δ vs 原生 K2 | 171 题复核 Δ vs 冻结的 K2 LoopCD |",
        "|---|---:|---:|---:|",
        *[f'| {field.upper()} | {oracle[f"{field}_teacher"]["vs_same_depth_native"]["delta_pp"]:+.2f} pp '
          f'| {hold[f"{field}_teacher"]["vs_same_depth_native"]["delta_pp"]:+.2f} pp '
          f'| {follow["vs_frozen_k2_baseline"][field]["delta_pp"]:+.2f} pp |' for field in ("q", "k", "v")],
        "",
        "Q 和 V 的净效应在复核中换向。所有分量均未通过预设门槛，复核中也均落后于验证集选定的浅轮 LoopCD。"
        "因此停止共享 Q LoRA、输出蒸馏和 Loop Dropout 分支，也没有生成可合并权重。"
        "这些移植使用额外原生 K4 教师计算。切取的 Q、K 已应用 RoPE，V 保持原生形式，同一候选前缀和位置参与比较。"
        "无操作钩子逐位一致。结果覆盖验证集上给定的四层窗口，对受控任务中的指定语义路径不作否定。",
        "",
        "## 运行与交付",
        "",
        f'RTX 4090 的共享测试扫描耗时 {test_run["wall_s"]:.2f} s，峰值分配 {test_run["peak_allocated_bytes"]/2**30:.2f} GiB。'
        "原生四档和冻选基线也独立跑完了全部测试候选，完整答案分数与共享扫描逐项相同。"
        "缓存测量使用按轮预分配槽位的 Transformers DynamicCache，绕开固定检查点默认缓存与 4.55.4 的兼容问题。"
        "本轮范围是单个 ARC-Challenge 数据集，Huginn、Ouro-2.6B 和跨任务泛化未运行。",
        "",
        "[完整基线表](BASELINES.md) · [逐题预测和成本](results/) · [实验记录](EXPERIMENT_LOG.md) · [复现](README.md)",
    ])
    write(dest / "EXPERIMENT_LOG.md", [
        "# Experiment log",
        "",
        "1. Pinned the official zero-shot ARC-C template and scored 299 validation questions, candidates in official order. Tested compact projection against the model wrapper at each depth. One sampled batch agreed on all per-token argmaxes and had a maximum 0.125 logit difference across BF16 GEMM shapes.",
        "2. Compared K4 Q/K/V teacher replacements with native and dose-matched controls in a predeclared four-layer window. The first hook setup selected a different mask path and failed parity. The delivered hook keeps the native SDPA configuration and passes bitwise identity.",
        "3. Screened clipped secant vs the validation-selected depth-matched references. Development acc_norm changed by "
        f'{config["secant_screen"]["3"]["delta_pp"]:+.2f} and {config["secant_screen"]["4"]["delta_pp"]:+.2f} pp at K3 and K4. '
        "Neither passed the gate. Froze source and scores before scoring test.",
        "4. Executed the complete 1,172-item test for all frozen variants. Followed up the oracle intervention on all 171 unused validation IDs with the same interventions and both controls. No oracle passed both development stages, and no training was triggered.",
        "5. Measured actual cached prefill and one-token decode. The checkpoint's default UniversalTransformerCache and a non-sized fallback failed against the installed Transformers. The recorded run explicitly sizes DynamicCache and checks the cache advances and the four native top token predictions match full recomputation.",
        "6. Measured standalone full-test wall time at each native depth and for the selected low-cost comparison. Archived scores, timings, resampling statistics, configuration hashes and non-promoted methods.",
    ])
    figure(norm, dest)
    sums = []
    for file in sorted(p for p in dest.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                       and p.name != "ARTIFACTS.sha256"):
        sums.append(f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(dest).as_posix()}")
    write(dest / "ARTIFACTS.sha256", sums)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    release(args.out, args.dest)
