"""Summarize matched ARC protocols, historical trajectories, and LoopCD decisions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from protocol_arc import CONTROLS, digest, dump

SCALES = ("raw", "character", "token")
DEPTHS = range(1, 5)
PROTOCOLS = tuple("ABCDEF")
COLORS = ("#4D4D4D", "#767676", "#3775BA", "#0F4D92", "#CC675E", "#A73C3C")
SYMBOLS = ("o", "s", "v", "^", "D", "P")
LINESTYLES = ("--", "-", "--", "-", "--", "-")


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def load(out, phase, split, condition, rotated=False):
    tag = f"{phase}.{split}.{condition}" + (".rotated" if rotated else "")
    path = out / f"{tag}.jsonl"
    execution = json.loads((out/f"{tag}.execution.json").read_bytes())
    assert digest(path.read_bytes()) == execution["scores_sha256"]
    records = lines(path)
    assert len({r["id"] for r in records}) == len(records) == execution["items"]
    return {r["id"]: r for r in records}, execution


def score_order(record, name, scale):
    values = np.array(record["scores"][name])
    if scale == "character":
        # A one-character letter target and a one-token text target are distinct.
        length = record["characters"] if record["target_format"] == "text" else 1
        values = values / np.array(length)
    if scale == "token":
        values = values / np.array(record["continuation_tokens"])
    order = np.array(record["order"])
    return int(order[np.argmax(values[order])]), int((values == values.max()).sum())


def annotate(records, condition):
    def target(c):
        return "letter" if c in ("E", "F", "F_text_demo") else "text"
    for r in records.values():
        r["target_format"] = target(condition)
    return records


def vector(records, name, scale, ids=None):
    ids = records if ids is None else ids
    pairs = [score_order(records[q], name, scale) for q in ids]
    return np.array([p for p, t in pairs]), np.array([t for p, t in pairs])


def track(hit, ids):
    assert hit.shape == (len(ids), 4)
    categories = Counter("".join(str(int(v)) for v in row) for row in hit)
    first = {"never": 0, **{str(k): 0 for k in DEPTHS}}
    first_then_regressed = {str(k): 0 for k in DEPTHS}
    regressions = []
    for identifier, row in zip(ids, hit):
        if not row.any():
            first["never"] += 1
            continue
        start = int(row.argmax())
        first[str(start+1)] += 1
        if not row[start:].all():
            regressions.append(identifier)
            first_then_regressed[str(start+1)] += 1
    transitions = {}
    for k, j in ((1, 2), (2, 3), (3, 4), (1, 4)):
        a, b = hit[:, k-1], hit[:, j-1]
        transitions[f"{k}->{j}"] = {
            "correct_to_wrong": int((a & ~b).sum()),
            "wrong_to_correct": int((~a & b).sum()),
            "correct_to_correct": int((a & b).sum()),
            "wrong_to_wrong": int((~a & ~b).sum()),
            "wrong_to_correct_ids": [q for q, selected in zip(ids, ~a & b) if selected],
            "correct_to_wrong_ids": [q for q, selected in zip(ids, a & ~b) if selected],
        }
    return {"n": len(ids), "correct": hit.sum(0).tolist(), "patterns": dict(categories),
            "first_correct": first, "first_correct_then_regressed": first_then_regressed,
            "regression_ids": regressions, "transitions": transitions}


def paired(a, b):
    n = len(a)
    assert n and len(b) == n
    gained, lost = int((~a & b).sum()), int((a & ~b).sum())
    rng = np.random.default_rng(20261120)
    counts = rng.multinomial(n, [(n-gained-lost)/n, gained/n, lost/n], size=9999)
    low, high = np.quantile((counts[:, 1] - counts[:, 2])/n, [.025, .975])
    return {"n": n, "correction": gained, "regression": lost, "delta_pp": (gained-lost)*100/n,
            "paired_bootstrap_95_pp": [float(low*100), float(high*100)],
            "exact_mcnemar_p": float(binomtest(gained, gained+lost).pvalue) if gained+lost else 1.0}


def gain_interaction(a,b):
    outcome = (b[:,3].astype(int) - b[:,0]) - (a[:,3].astype(int) - a[:,0])
    values, counts = np.unique(outcome, return_counts=True)
    draws = np.random.default_rng(20261120).multinomial(len(outcome), counts/len(outcome), size=9999)
    confidence = np.quantile(draws @ values * 100 / len(outcome), [.025,.975]).tolist()
    return {"delta_pp": float(outcome.mean()*100), "paired_bootstrap_95_pp": confidence}


def get_native(records, scale, ids):
    preds, ties = zip(*(vector(records, f"native_k{k}", scale, ids) for k in DEPTHS))
    return np.column_stack(preds), np.column_stack(ties)


def comparable(a, b, same_prompt=False):
    ids = list(q for q in a if q in b)
    assert ids and all(a[q]["gold"] == b[q]["gold"] and a[q]["labels"] == b[q]["labels"] for q in ids)
    if same_prompt:
        assert all(a[q]["context_ids_sha256"] == b[q]["context_ids_sha256"] for q in ids)
    return ids


def protocol_native(natives):
    metrics = {}
    for condition, records in natives.items():
        ids = list(records)
        gold = np.array([records[q]["gold"] for q in ids])
        lengths = np.array([records[q]["context_tokens"] for q in ids])
        certain = [records[q]["content_tokens"] for q in ids]
        char_lengths = [records[q]["characters"] for q in ids]
        min_token = np.array([l[g] == min(l) for l,g in zip(certain, gold)])
        max_token = np.array([l[g] == max(l) for l,g in zip(certain, gold)])
        char_unique_longest = np.array([l[g] == max(l) and l.count(l[g]) == 1 for l,g in zip(char_lengths, gold)])
        char_unique_shortest = np.array([l[g] == min(l) and l.count(l[g]) == 1 for l,g in zip(char_lengths, gold)])
        length_bin = {"gold_min_token_including_ties": min_token, "gold_max_token_including_ties": max_token,
                      "gold_unique_min_char": char_unique_shortest, "gold_unique_max_char": char_unique_longest}
        out = {"context_tokens_min_median_max": [int(lengths.min()), float(np.median(lengths)), int(lengths.max())],
               "scales": {}}
        for scale in SCALES:
            pred, ties = get_native(records, scale, ids)
            hit = pred == gold[:,None]
            t = track(hit, ids)
            t["top_score_ties"] = (ties > 1).sum(0).tolist()
            t["length_strata"] = {b: {"n": int(m.sum()), "correct": hit[m].sum(0).tolist()}
                                  for b,m in length_bin.items()}
            out["scales"][scale] = t
        metrics[condition] = out
    return metrics


def native_contrasts(natives, rotated):
    pairs = (
        ("A","B","0→25-shot open/text",False),
        ("A","C","show options 0-shot/text",False),
        ("C_mask_options","C","show options fixed positions 0-shot/text",True),
        ("D_mask_options","D","show options fixed positions 25-shot/text",True),
        ("C","D","0→25-shot shown/text",False),
        ("E","F","0→25-shot shown/letter",False),
        ("C","E","text→letter same prompt 0-shot",True),
        ("D","F_text_demo","text→letter same prompt 25-shot text demos",True),
        ("D_letter_demo","F","text→letter same prompt 25-shot letter demos",True),
        ("D","D_letter_demo","text→letter demo answers with text target",False),
        ("F_text_demo","F","text→letter demo answers with letter target",False),
        ("B","D","show options in demos and query 25-shot/text",False),
    )
    metrics = {}
    for ca, cb, name, same_prompt in pairs:
        a, b = natives[ca], natives[cb]
        ids = comparable(a, b, same_prompt)
        if "mask_options" in ca:
            assert all(a[q]["context_tokens"] == b[q]["context_tokens"] for q in ids)
            assert all(a[q]["masked_positions"] is not None and b[q]["masked_positions"] is None for q in ids)
        gold = np.array([a[q]["gold"] for q in ids])[:,None]
        out = {}
        for scale in SCALES:
            pl = get_native(a, scale, ids)[0] == gold
            pr = get_native(b, scale, ids)[0] == gold
            out[scale] = {str(k): paired(pl[:,k-1], pr[:,k-1]) for k in DEPTHS}
            interaction = gain_interaction(pl,pr)
            out[scale]["change_in_K1_to_K4_gain_pp"] = interaction["delta_pp"]
            out[scale]["change_in_K1_to_K4_gain_paired_bootstrap_95_pp"] = interaction["paired_bootstrap_95_pp"]
        metrics[name] = {"baseline": ca, "comparison": cb, "scales": out}
    for c, b in rotated.items():
        a = natives[c]
        ids = comparable(a,b)
        gold = np.array([a[q]["gold"] for q in ids])[:,None]
        assert all(a[q]["order"] != b[q]["order"] for q in ids)
        out = {}
        for scale in SCALES:
            pl = get_native(a, scale, ids)[0]
            pr = get_native(b, scale, ids)[0]
            out[scale] = {str(k): paired(pl[:,k-1] == gold[:,0], pr[:,k-1] == gold[:,0]) for k in DEPTHS}
            out[scale]["same_predicted_content"] = (pl==pr).sum(0).tolist()
        metrics[f"{c} rotated candidate order"] = {"baseline": c, "comparison": c+" rotated", "scales": out}
    return metrics


def compare_loopcd(natives, loopcd):
    metrics = {}
    for c in PROTOCOLS:
        a, b = natives[c], loopcd[c]
        ids = comparable(a, b, same_prompt=True)
        assert len(a) == len(b) == len(ids)
        gold = np.array([a[q]["gold"] for q in ids])
        for scale in SCALES:
            native = get_native(a, scale, ids)[0] == gold[:,None]
            out = {}
            for method in sorted(b[ids[0]]["scores"]):
                k = int(method[-1])
                orig = native[:,k-1]
                pred, ties = vector(b, method, scale, ids)
                with_cd = pred == gold
                entry = {"correct": int(with_cd.sum()), "ties": int((ties>1).sum()), **paired(orig, with_cd)}
                fixes = ~orig & with_cd
                entry["corrected_ids"] = [q for q,m in zip(ids, fixes) if m]
                entry["corrected_with_native_k1_correct"] = int((fixes & native[:,0]).sum())
                entry["corrected_after_all_native_budgets_so_far_wrong"] = int(
                    (fixes & ~native[:,:k].any(1)).sum())
                if k < 4:
                    later = native[:,k:]
                    future = ~orig & later.any(1)
                    next_fix = ~orig & later[:,0]
                    entry.update({
                        "native_next_corrected": int(next_fix.sum()),
                        "both_this_method_and_native_next_corrected": int((fixes & next_fix).sum()),
                        "native_any_later_corrected": int(future.sum()),
                        "both_this_method_and_native_any_later_corrected": int((fixes & future).sum()),
                        "native_later_only_ids": [q for q,m in zip(ids, future & ~fixes) if m],
                        "loopcd_only_ids": [q for q,m in zip(ids, fixes & ~future) if m],
                    })
                out[method] = entry
            metrics.setdefault(c,{})[scale] = out
    return metrics


def historical(path):
    source = json.loads(path.read_bytes())
    rows = source["question_trajectories"]
    assert len(rows) == len({q["id"] for q in rows}) == 1172
    assert not any(q["is_demonstration"] or q["demonstration_question_text_overlap"] for q in rows)
    ids = [q["id"] for q in rows]
    gold = np.array([q["gold"] for q in rows])
    hit = np.asarray([q["predictions"] for q in rows]) == gold[:,None]
    summary = track(hit, ids)
    assert summary["correct"] == source["all"]["correct_counts"]
    return {"source_sha256": digest(path.read_bytes()), "source_accuracy_checked": summary["correct"], "archive": summary}


def figures(out, native, contrasts, loopcd):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.25, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained", sharey=True)
    for ax, scale in zip(axes, ("raw","character")):
        for c,color,symbol,style in zip(PROTOCOLS, COLORS, SYMBOLS, LINESTYLES):
            r = native[c]["scales"][scale]
            ax.plot(list(DEPTHS), np.array(r["correct"])*100/r["n"], marker=symbol, linestyle=style,
                    color=color, label=c, linewidth=1.7, markersize=5)
        ax.set(xlabel="Native loops", xticks=list(DEPTHS), title="Raw log likelihood" if scale=="raw"
               else "Character-normalized log likelihood")
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Accuracy (%) · ARC-Challenge validation, n = 299")
    axes[1].legend(title="Protocol", frameon=False, loc="best", ncol=2)
    for ext in ("png","pdf"):
        fig.savefig(out/f"figure1_protocol_accuracy.{ext}", dpi=300)
    plt.close(fig)
    if loopcd:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained", sharey=True)
        for ax,k in zip(axes,(2,3,4)):
            base = np.arange(len(PROTOCOLS))
            groups = zip(("fixed","adaptive","hidden"),("#0F4D92","#42949E","#B64342"),("o","s","D"))
            for i,(m,color,marker) in enumerate(groups):
                rows = [loopcd[c]["character"][f"loopcd_{m}_k{k}"] for c in PROTOCOLS]
                values = np.array([r["delta_pp"] for r in rows])
                low, high = np.array([r["paired_bootstrap_95_pp"] for r in rows]).T
                ax.errorbar(base+(i-1)*.19, values, yerr=[values-low,high-values], fmt=marker, markersize=5,
                            color=color, label=m.capitalize(), capsize=2, linewidth=1)
            ax.axhline(0,color="#909090",linewidth=.8)
            ax.set(title=f"K{k}", xlabel="ARC protocol", xticks=base, xticklabels=PROTOCOLS)
            ax.grid(axis="y",alpha=.20)
        axes[0].set_ylabel("Accuracy change vs native at the same budget (pp)")
        axes[-1].legend(frameon=False, loc="best")
        for ext in ("png","pdf"):
            fig.savefig(out/f"figure7_loopcd_effects.{ext}", dpi=300)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("historical", type=Path)
    p.add_argument("--loopcd", action="store_true")
    a = p.parse_args()
    frozen = json.loads((a.out/"PROTOCOL_FREEZE.json").read_bytes())
    phases = {}
    for split in ("train","validation"):
        native, rotated, timings = {}, {}, {}
        for c in sorted(set(PROTOCOLS) | CONTROLS):
            records, cost = load(a.out,"native",split,c)
            expected = frozen["selection"][split]["controls" if c in CONTROLS else "main"]
            assert set(records) == set(expected)
            native[c], timings[c] = annotate(records,c), cost
            if c in "CDEF":
                r, cost = load(a.out,"native",split,c,True)
                assert set(r) == set(frozen["selection"][split]["controls"])
                rotated[c], timings[c+".rotated"] = annotate(r,c), cost
        loopcd = {}
        if a.loopcd:
            for c in PROTOCOLS:
                records, cost = load(a.out,"loopcd",split,c)
                assert set(records) == set(frozen["selection"][split]["main"])
                loopcd[c], timings[c+".loopcd"] = annotate(records,c), cost
        phases[split] = {
            "native": protocol_native(native),
            "contrasts": native_contrasts(native,rotated),
            "loopcd": compare_loopcd(native,loopcd) if loopcd else {},
            "execution": timings,
        }
    result = phases["validation"]
    name = "PROTOCOL_METRICS_WITH_LOOPCD.json" if a.loopcd else "PROTOCOL_NATIVE_METRICS.json"
    dump(a.out/name, {**phases, "historical_descriptive_only": historical(a.historical)})
    figures(a.out, result["native"], result["contrasts"], result["loopcd"])
    print(name, {c: result["native"][c]["scales"]["character"]["correct"] for c in PROTOCOLS})


if __name__ == "__main__":
    main()
