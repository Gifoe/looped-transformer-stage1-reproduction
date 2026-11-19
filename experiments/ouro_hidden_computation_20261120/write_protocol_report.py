"""Render the protocol report from verified experiment and execution outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def compact_agreement(result, protocol):
    compact = {"B": "A", "D": "C", "F": "E"}[protocol]
    shared = {r["id"]: r for r in read_rows(result/f"native.validation.{compact}.jsonl")}
    masked = read_rows(result/f"native.validation.{protocol}_mask_demonstrations.jsonl")
    def choice(r, k):
        z = r["scores"][f"native_k{k}"]
        length = r["characters"] if protocol != "F" else [1] * len(z)
        return max(r["order"], key=lambda c: z[c]/length[c])
    assert all(shared[r["id"]]["gold"] == r["gold"] for r in masked)
    return [sum(choice(r,k) == choice(shared[r["id"]],k) for r in masked) for k in range(1,5)]


def chain(numbers):
    return "/".join(str(v) for v in numbers)


def pair(x):
    low, high = x["paired_bootstrap_95_pp"]
    return f'{x["delta_pp"]:+.2f} [{low:+.2f}, {high:+.2f}]'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    a = p.parse_args()
    result = a.results
    data = read(result/"PROTOCOL_METRICS_WITH_LOOPCD.json")
    generation = read(result/"GENERATION_COST_METRICS.json")
    timing_audit = read(result/"PROTOCOL_GENERATION_AUDIT.json")
    demonstration_mask = read(result/"DEMONSTRATION_MASK_METRICS.json")
    valid = data["validation"]
    native, contrasts, loopcd = (valid[v] for v in ("native","contrasts","loopcd"))
    hist = data["historical_descriptive_only"]["archive"]
    matched = contrasts["text→letter same prompt 0-shot"]["scales"]["character"]
    visibility = contrasts["show options fixed positions 25-shot/text"]["scales"]["character"]
    open_correct = native["B"]["scales"]["character"]["correct"]
    shown_correct = native["D"]["scales"]["character"]["correct"]
    lines = [
        "# Ouro-1.4B Base 的六协议匹配比较",
        "",
        f"在 25-shot 全文评分下，无查询候选的 B 从 {open_correct[0]} 升到 {open_correct[3]} 题。展示候选的 D 从 {shown_correct[0]} 升到 {shown_correct[3]} 题。同样长度和查询坐标的 25-shot 遮蔽对照估计 K4 候选可见性的变化为 {pair(visibility['4'])} 个百分点（n={visibility['4']['n']}）。在完全相同的 0-shot 提示下，全文评分切换到字母评分，K4 的变化为 {pair(matched['4'])} 个百分点。前一个比较改变模型的可见输入，后一个比较改变候选续写评分规则。",
        "",
        "验证集为 ARC-Challenge 官方 Validation 全部 299 题，此数据在既有研究中用于评测。开发集为官方 Train 中按固定哈希选出的 64 题，排除了 25 个历史示例及与 Validation 题干规范化后重复的题目。六个主条件使用相同候选内容、题干和正确答案。模型冻结在修订 `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`，BF16、SDPA、RTX 4090。六协议分别抽查 K1–K4，官方独立深度前向与统一轨迹的词表 logits 逐项相等。A 的 Validation 分数复用提交 `5557699` 的相同输入和公式，读取时验证题号、选项标签、正确答案、字符数、续写 token 数与实际输入 token 数。",
        "",
        "每个候选的原始分数为全词表 FP32 log-softmax 的续写 token 总和。字符归一化除以续写字符数，不计连接空格，字母分母为 1。另计算 token 归一化、金答案长度分层和精确并列。并列以展示顺序首次出现为准。所有结果的分母由表头或 n 列给出。",
        "",
        "## 验证集原生轨迹",
        "",
        "| 协议 | 示例 / 查询候选 / 目标 | 输入 token 最小/中位数/最大 | 原始正确 K1/K2/K3/K4 | 字符归一化正确 K1/K2/K3/K4 |",
        "|---|---|---:|---:|---:|",
    ]
    description = {
        "A": "0 / 隐藏 / 全答案", "B": "25 / 隐藏 / 全答案",
        "C": "0 / 展示 / 全答案", "D": "25 / 展示 / 全答案",
        "E": "0 / 展示 / 字母", "F": "25 / 展示 / 字母",
    }
    for c in "ABCDEF":
        m = native[c]
        lengths = [int(v) for v in m["context_tokens_min_median_max"]]
        lines.append(f'| {c} | {description[c]} | {chain(lengths)} | {chain(m["scales"]["raw"]["correct"])} | {chain(m["scales"]["character"]["correct"])} |')
    lines.extend([
        "",
        "![六协议在四个原生循环深度下的准确率](protocol_results/figure1_protocol_accuracy.png)",
        "",
        "| 协议 | W→C K1→2 / 2→3 / 3→4 | C→W K1→2 / 2→3 / 3→4 | 首次正确 K1/K2/K3/K4/从未 | 首次正确后出现回退 |",
        "|---|---:|---:|---:|---:|",
    ])
    for c in "ABCDEF":
        m = native[c]["scales"]["character"]
        t = [m["transitions"][k] for k in ("1->2","2->3","3->4")]
        first = [m["first_correct"][k] for k in ("1","2","3","4","never")]
        lines.append(f'| {c} | {chain(x["wrong_to_correct"] for x in t)} | {chain(x["correct_to_wrong"] for x in t)} | {chain(first)} | {sum(m["first_correct_then_regressed"].values())} |')
    lines.extend([
        "",
        "## 匹配对照",
        "",
        "C 与 E 的查询 token 完全相同，差异来自对同一前缀的续写评分和结果解析。25-shot 查询的示例答案和测试目标采用交叉条件。C、D 的遮蔽条件保留候选区域的 token 数、答案位置与 RoPE 坐标。候选区域配成同质 token 并从每层注意力中排除。遮蔽条件固定名义长度，减少了有效可见上下文。25-shot 完整示例与 0-shot 的对比包括示例内容和有效上下文的变化。候选重新排序只用每题固定的一个非零循环移位，预测后映射到原候选内容。",
        "",
        "下表为字符归一化准确率变化，单位百分点，附逐题配对重采样 95% 区间。K4−K1 一列比较这段增益在两种条件之间的差值。区间逐对计算。",
        "",
        "| 对照（先前→之后） | n | K1 | K4 | K4−K1 增益差 |",
        "|---|---:|---:|---:|---:|",
    ])
    chinese = {
        "0→25-shot open/text": "A→B 无候选 / 全答案",
        "show options 0-shot/text": "A→C 展示查询候选",
        "show options fixed positions 0-shot/text": "C 遮蔽→C 0-shot，固定位置",
        "show options fixed positions 25-shot/text": "D 遮蔽→D 25-shot，固定位置",
        "0→25-shot shown/text": "C→D 展示候选 / 全答案",
        "0→25-shot shown/letter": "E→F 展示候选 / 字母",
        "text→letter same prompt 0-shot": "C→E 相同输入，目标由全文到字母",
        "text→letter same prompt 25-shot text demos": "D→F 交叉，示例为全文答复",
        "text→letter same prompt 25-shot letter demos": "D 交叉→F，示例为字母答复",
        "text→letter demo answers with text target": "D→D 交叉，全文目标，改变示例答复",
        "text→letter demo answers with letter target": "F 交叉→F，字母目标，改变示例答复",
        "show options in demos and query 25-shot/text": "B→D 示例和查询均增加候选",
    }
    for key, title in chinese.items():
        m = contrasts[key]["scales"]["character"]
        lines.append(f'| {title} | {m["1"]["n"]} | {pair(m["1"])} | {pair(m["4"])} | {m["change_in_K1_to_K4_gain_pp"]:+.2f} |')
    lines += ["", "| 候选次序变化 | n | 两种顺序下预测同一内容 K1/K2/K3/K4 | K4 准确率差 |",
              "|---|---:|---:|---:|"]
    for c in "CDEF":
        m = contrasts[f"{c} rotated candidate order"]["scales"]["character"]
        lines.append(f'| {c} | {m["1"]["n"]} | {chain(m["same_predicted_content"])} | {pair(m["4"])} |')
    lines += [
        "",
        "另在同一组固定的 32 题上，从注意力可见范围移除全部示例。比较双方保留完全相同的 query token、提示总长度、query 位置及评分目标。这个对照检验相同名义长度下，能否读到示例对准确率的影响。",
        "",
        "| 协议 | 遮蔽前缀 token 数 | 可见示例正确 K1/K2/K3/K4 | 遮蔽示例正确 K1/K2/K3/K4 | 遮蔽与实际 0-shot 的预测一致数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for protocol, item in demonstration_mask.items():
        lines.append(
            f'| {protocol} | {item["masked_prefix_tokens"]} | '
            f'{chain(item["same_length_unmasked_correct"])} | {chain(item["masked_correct"])} | '
            f'{chain(compact_agreement(result,protocol))} |')
    lines += [
        "",
        "## 历史 Test 描述性复核",
        "",
        "历史 25-shot Joint-Option 的另一份 A6000 运行已经归档所有 1,172 道题的 K1–K4 预测。这四个预算采用同一运行的逐题轨迹。早期原版公开端点为 634/1172 和 1021/1172。两份历史运行有 36 道题的端点预测发生变化，具体差异见早期的 `historical_joint_comparison.json`。历史 Test 专用于描述性复核。",
        "",
        f'全轨迹归档正确 K1/K2/K3/K4 为 **{chain(hist["correct"])} / 1172**。首次正确 K1/K2/K3/K4 和从未正确为 {chain(hist["first_correct"][v] for v in ("1","2","3","4","never"))}。首次正确后有 {sum(hist["first_correct_then_regressed"].values())} 题回退过。',
        "",
        "| 深度变化 | W→C | C→W |",
        "|---|---:|---:|",
    ]
    for key,t in hist["transitions"].items():
        lines.append(f'| {key} | {t["wrong_to_correct"]} | {t["correct_to_wrong"]} |')
    lines += [
        "",
        "## 同协议、同预算的 LoopCD",
        "",
        "Fixed 使用最终 logits 加 0.5 倍最终与 K1 的差。Adaptive 依据最终全词表概率前两名的差调节对比系数。Hidden 在读出前混合当轮和 K1 的 Norm 前状态，再运行模型 Norm 和词表头。三个变体使用现有原生循环，教师强制评分时干预均发生在读出，不回写循环状态。K1 没有可对比的早轮，以 Native K1 为共同基准。无约束生成时，读出改选的 token 可以成为下一解码步的新输入。",
        "",
        "下表列出 Validation 299 题的字符归一化正确数。完整逐题结果附原始和 token 归一化分数。",
        "",
        "| 协议 | K2 Native/Fixed/Adaptive/Hidden | K3 Native/Fixed/Adaptive/Hidden | K4 Native/Fixed/Adaptive/Hidden |",
        "|---|---:|---:|---:|",
    ]
    for c in "ABCDEF":
        rows = loopcd[c]["character"]
        nc = native[c]["scales"]["character"]["correct"]
        columns = [chain([nc[k-1]]+[rows[f"loopcd_{m}_k{k}"]["correct"] for m in ("fixed","adaptive","hidden")])
                   for k in (2,3,4)]
        lines.append(f'| {c} | ' + " | ".join(columns) + " |")
    lines += [
        "",
        "![不同协议下 LoopCD 相对同预算原生循环的变化](protocol_results/figure7_loopcd_effects.png)",
        "",
        "以下固定报告 K3 的 Fixed 对比。以 Native K3 错题为风险集，计算读出纠正与 Native K4 纠正的交集。其余变体及 K2 对比见机器可读结果。",
        "",
        "| 协议 | Fixed K3 纠正 | 其中 Native K1–K3 全错 | Native K4 纠正 | 交集 | 仅 Native K4 纠正 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in "ABCDEF":
        row = loopcd[c]["character"]["loopcd_fixed_k3"]
        lines.append(f'| {c} | {row["correction"]} | {row["corrected_after_all_native_budgets_so_far_wrong"]} | {row["native_next_corrected"]} | {row["both_this_method_and_native_next_corrected"]} | {len(row["native_later_only_ids"])} |')
    n = generation["A"]["native_k1"]["rows"]
    lines += [
        "",
        "## 候选评分时间和实际生成",
        "",
        f"在预先哈希选取的同一批 {n} 道 Validation 题上，独立执行全部协议和相应深度的变体。评分按一题一批，批内按完全相同的输入去重并右填充。生成使用无约束词表贪心，有原生 KV cache，遇到换行或 EOS 停止。只匹配首行的候选全文或独立字母，生成替代表达未做语义判分。以下为 K4 的示例表，所有深度的实际数据都可查询。",
        "",
        "| 协议 | K4 生成正确 Native/Fixed/Adaptive/Hidden | 候选评分中位 ms | 生成 prefill 中位 ms |",
        "|---|---:|---:|---:|",
    ]
    for c in "ABCDEF":
        rows = [generation[c][f"native_k4"]] + [generation[c][f"loopcd_{m}_k4"] for m in ("fixed","adaptive","hidden")]
        candidate_ms = chain(format(r["median_candidate_ms"], ".1f") for r in rows)
        prefill_ms = chain(format(r["median_prefill_ms"], ".1f") for r in rows)
        lines.append(f'| {c} | {chain(r["exact_correct"] for r in rows)} | {candidate_ms} | {prefill_ms} |')
    matching_generated_rows = 0
    total_generation_rows = 0
    for method in generation["C"]:
        source, target = [read(result/f"generation.{c}.{method}.json") for c in ("C","E")]
        assert [r["id"] for r in source] == [r["id"] for r in target]
        assert [r["prompt_sha256"] for r in source] == [r["prompt_sha256"] for r in target]
        total_generation_rows += len(source)
        matching_generated_rows += sum(x["generated_ids"]==y["generated_ids"] for x,y in zip(source,target))
    score_disagreements = sum(
        sum(v["standalone_vs_batched_character_normalized_prediction_disagreements"].values())
        for v in timing_audit["candidate_score_agreement"].values()
    )
    lines += [
        "",
        f"C 与 E 的相同提示对在 {matching_generated_rows}/{total_generation_rows} 个题目 × 方法条件下生成逐 token 完全相同的续写。两种协议按各自目标格式解析这些输出。",
        "",
        "全文候选评分运行成本包含各候选续写的 token work。字母候选共用同一条前向轨迹。批处理评分时间和逐题生成时间分别记录，批处理阶段与因果实验共享 GPU。逐题生成和候选评分诊断在 GPU 空闲后启动。",
        f"统一 K1–K4 轨迹的批处理评分与逐题独立深度计时评分有 {score_disagreements}/{timing_audit['generation_rows']} 条字符归一化候选预测不一致。六组主准确率采用统一批评分。逐题执行用于延迟和生成诊断。",
        "",
        "## 可追溯结果",
        "",
        "- [冻结配置、样本 ID、输入和代码哈希](protocol_results/PROTOCOL_FREEZE.json)",
        "- [固定名义长度的示例遮蔽对照](protocol_results/DEMONSTRATION_MASK_METRICS.json)",
        "- [验证集、开发集、历史测试轨迹、配对检验、并列及原始结果的 SHA-256](protocol_results/PROTOCOL_METRICS_WITH_LOOPCD.json)",
        "- [每个协议和方法的候选评分、生成、prefill、decode 执行记录](protocol_results/GENERATION_COST_METRICS.json)",
        "- [逐题执行、候选预测及生成的核查结果](protocol_results/PROTOCOL_GENERATION_AUDIT.json)",
        "- [逐题候选分数和生成输出](protocol_results/)",
        "",
    ]
    (result.parent/"PROTOCOL_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
