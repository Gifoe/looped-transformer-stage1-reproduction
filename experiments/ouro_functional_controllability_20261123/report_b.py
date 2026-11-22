"""Render the frozen controller and full candidate/generation comparisons."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def seq(items):
    return "/".join(str(v) for v in items)


def run(root, out):
    freeze = read(root/"FREEZE.json")
    bench = read(root/"BENCHMARK_RESULTS/BENCHMARK.json")
    costs = read(root/"BENCHMARK_RESULTS/COST.json")
    screen = read(root/"VAL_SCREEN.json")
    generation = read(root/"GENERATION_RESULTS/GENERATION_SUMMARY.json")
    compute = read(root/"GENERATION_RESULTS/COST.json")
    gate = read(root/"SEVEN_TASK_GATE.json")
    methods = ("native","linear8_t1","linear8_t2",*freeze["weights"])
    main = freeze["primary"]
    lines = [
        "# Experiment B · 2K–4K 共享控制律",
        "",
        f"冻结方法为 `{main}`，{'通过' if freeze['selection_eligible'] else '未通过'} ARC Validation 生成选型门槛。"
        "训练只用 ARC Train，Test 只评估锁定方法。",
        "",
        "| ARC Validation | 可训练／保存参数 | 更新／后缀调用 | acc K1–K4 | acc_norm K1–K4 | K2–K4 生成正确率 | 有效率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method,item in screen.items():
        f,c,g = item["fit"],item["candidate"],item["generation"]
        lines.append(f"| `{method}` | {f['trainable_parameters']:,}/{f['stored_scalars']:,} | "
                     f"{f['steps']}/{f['network_calls']} | {seq(c['acc'])} | {seq(c['acc_norm'])} | "
                     f"{100*g['correct']:.2f}% | {100*g['valid']:.2f}% |")
    for label, group in bench.items():
        data=group["summary"]
        lines += ["",f"## {label} Test，{data['n']:,} 题","",
                  "| 方法 | acc 正确数 K1–K4 | acc_norm 正确数 K1–K4 | 批中位延迟 ms K1–K4 |",
                  "|---|---:|---:|---:|"]
        for m in methods:
            counts=[seq(data["metrics"][k]["methods"][f"{m}_k{n}"]["correct"] for n in range(1,5))
                    for k in ("acc","acc_norm")]
            t=(seq(f"{costs[label][m][f'k{k}']['median_batch_ms']:.2f}" for k in range(1,5))
               if m in costs[label] else "历史")
            lines.append(f"| `{m}` | {counts[0]} | {counts[1]} | {t} |")
        if label=="ARC":
            norm=data["metrics"]["acc_norm"]
            h=norm["top_loopcd"]
            counts=seq(norm["methods"][h[f"k{k}"]]["correct"] for k in (2,3,4))
            lines += ["",f"同预算最佳历史 LoopCD 的 K2–K4 `acc_norm` 答对数为 {counts}。"]
        lines.append("")
        for metric,p in group["primary"].items():
            lo,hi=p["ci95_pp"]
            gain=p["gain_retention"]
            lines += [f"主方法 K2–K4 `{metric}` 相对 Rank-8 T1 净变化 {p['net_pp']:+.2f} pp "
                      f"[{lo:+.2f}, {hi:+.2f}]，救回／退化 {p['rescue']}/{p['damage']}，"
                      f"相对 Native 的 Rank-8 增益保留 {100*gain:.1f}%。"]
    lines += ["","## 真实贪心生成"]
    for ds,d in generation["by_depth"].items():
        lines += ["", f"### {ds}", "",
                  "| 模型 | 答对 K1–K4 | 有效 K1–K4 | 新运行时间秒 K1–K4 |",
                  "|---|---:|---:|---:|"]
        for m in methods:
            four=[d[f"k{k}"][m] for k in range(1,5)]
            timing = (seq(f"{compute[f'{ds}__{m}__k{k}']['wall_s']:.1f}" for k in range(1,5))
                      if m in freeze["weights"] else "历史")
            lines.append(f"| `{m}`（{four[0]['n']}） | {seq(x['correct'] for x in four)} | "
                         f"{seq(x['valid'] for x in four)} | {timing} |")
        z=generation["paired_with_prior"][ds][main]
        p,q=z["correct"],z["valid"]
        lines+=["",f"{ds} 的主方法对预先确定的生成参照 `{generation['pooled_reference']}`，"
                f"K2–K4 正确率净差 {p['net_pp']:+.2f} pp，有效率净差 {q['net_pp']:+.2f} pp。"]
    lines+=["", "按题聚类的区间、单轮救回和退化、无效输出类型、生成 token 分布与真实显存和时间均在对应结果目录。"
                "历史候选分数与历史贪心结果直接复用。ARC 贪心报告使用 Validation，OpenBookQA 使用独立 Test。",
             f"八个预注册检查中通过 {sum(gate['checks'])} 个。七项扩展的性能门槛{'通过' if gate['passed'] else '未通过'}。",
             " [公式与参数量](METHOD.md) [事前门槛](THEORY_HYPOTHESES.md)"]
    out.write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("report",type=Path)
    a=p.parse_args()
    run(a.results,a.report)
