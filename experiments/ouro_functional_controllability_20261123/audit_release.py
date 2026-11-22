"""Verify completeness and headline counts from the released evidence without a GPU."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as rows:
        return [json.loads(line) for line in rows]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_ledger(path):
    jobs = []
    total = 0
    opened = None
    previous_end = 0
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        event, job = fields[:2]
        start = int(fields[2])
        if event == "START":
            assert opened is None and start >= previous_end, fields
            assert int(fields[3]) == 28800 - total, fields
            opened = job, start
            continue
        assert event == "END" and opened == (job, start), fields
        end, duration, rc = map(int, fields[3:])
        assert end - start == duration >= 0 and rc == 0, fields
        jobs.append(dict(job=job, seconds=duration))
        total += duration
        previous_end = end
        opened = None
    assert opened is None and len(jobs) == 11 and total <= 28800
    return dict(total_seconds=total, jobs=jobs)


def verify(root, results, bench, generation):
    freeze = read_json(results / "FREEZE.json")
    gate = read_json(results / "SEVEN_TASK_GATE.json")
    assert freeze["primary"] == "rank1_t1" and freeze["selection_eligible"]
    assert gate["passed"] is False
    assert freeze["predeclared_theory_sha256"] == sha256(root / "THEORY_HYPOTHESES.md")
    assert len(freeze["weights"]) == 3 and sum(gate["checks"]) == 7
    screen = read_json(results / "VAL_SCREEN.json")
    weights = {path.stem: path for path in (results / "weights").glob("*.pt")}
    assert len(weights) == 8 and set(weights) == set(screen)
    checkpoint_index = {key: dict(file=path.name, sha256=sha256(path), fit=screen[key]["fit"])
                        for key, path in sorted(weights.items())}
    for key, checkpoint in freeze["weights"].items():
        path = results / "weights" / checkpoint["file"]
        assert checkpoint_index[key]["sha256"] == checkpoint["sha256"], path

    benchmark_counts = {}
    benchmarks = {"ARC": (1172, [417, 721, 764, 760], [442, 758, 806, 797]),
                  "OPENBOOKQA": (500, [95, 189, 206, 230], [168, 244, 270, 290])}
    compared = ("native", "linear8_t1", "linear8_t2", "rank1_t1", "fixed35", "smooth_t2")
    for dataset, (n, expected_acc, expected_norm) in benchmarks.items():
        rows = read_jsonl(bench / f"RESULTS_{dataset}.jsonl")
        assert len(rows) == len({row["id"] for row in rows}) == n
        metrics = {}
        for metric in ("acc", "acc_norm"):
            counts = defaultdict(list)
            for name in compared:
                for k in range(1, 5):
                    key = f"{name}_k{k}"
                    assert all(len(row["scores"][key]) == len(row["labels"]) for row in rows)
                    counts[name].append(sum(row["predictions"][metric][key] == row["gold"] for row in rows))
            assert counts["rank1_t1"] == (expected_acc if metric == "acc" else expected_norm)
            metrics[metric] = counts
        benchmark_counts[dataset] = dict(n=n, **metrics)

    grouped = defaultdict(Counter)
    ids = defaultdict(set)
    for row in read_jsonl(generation / "GENERATION_RESULTS.jsonl"):
        key = row["split"], row["method"], row["depth"]
        assert row["id"] not in ids[key], key
        ids[key].add(row["id"])
        counts = grouped[key]
        counts["n"] += 1
        counts["correct"] += int(row["correct"])
        counts["valid"] += int(row["valid"])
    gen_counts = {}
    sizes = {"arc_validation": 299, "openbookqa_test": 500}
    assert len(grouped) == len(sizes) * len(compared) * 4
    for split, n in sizes.items():
        for method in compared:
            for k in range(1, 5):
                key = split, method, k
                assert grouped[key]["n"] == n and ids[key] == ids[split, "native", 1], key
                gen_counts[f"{split}__{method}__k{k}"] = dict(grouped[key])
    expected = {"arc_validation": [129, 220, 248, 245], "openbookqa_test": [160, 325, 380, 387]}
    for split in sizes:
        assert [grouped[split, "rank1_t1", k]["correct"] for k in range(1, 5)] == expected[split]

    transfer = read_jsonl(results / "CSQA_TRANSFER/PREDICTIONS.jsonl")
    assert len(transfer) == len({row["id"] for row in transfer}) == 1221
    assert all(len(row["scores"]) == 12 for row in transfer)
    transfer_counts = {name: {metric: [] for metric in ("acc", "acc_norm")}
                       for name in ("native", "rank8", "frozen")}
    for name, metrics in transfer_counts.items():
        for metric, counts in metrics.items():
            for k in range(1, 5):
                counts.append(sum(row["predictions"][f"{name}_k{k}"][metric] == row["gold"] for row in transfer))
    semantic = read_jsonl(results / "E_SEMANTIC/PREDICTIONS.jsonl")
    assert len(semantic) == len({row["id"] for row in semantic}) == 1536
    assert read_json(results / "E_SEMANTIC/SUMMARY.json")["fixed_axis_semantic_specificity_gate"]["criterion_met"] is False

    return dict(ledger=inspect_ledger(root / "GPU_LEDGER.tsv"), primary=freeze["primary"],
                checkpoints=checkpoint_index, benchmark=benchmark_counts, generation=gen_counts,
                commonsenseqa_validation=transfer_counts, semantic_matched_rows=len(semantic),
                conditional_seven_task_gate=gate["passed"])


def verify_doc_links(root):
    for path in root.glob("*.md"):
        for link in re.findall(r"\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            if link.startswith(("http://", "https://", "../", "#")):
                continue
            assert (root / link.partition("#")[0]).exists(), (path.name, link)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    results = root / "results"
    bench = root / "BENCHMARK_RESULTS"
    if not bench.exists():
        bench = results / "BENCHMARK_RESULTS"
    generation = root / "GENERATION_RESULTS"
    if not generation.exists():
        generation = results / "GENERATION_RESULTS"
    if bench.parent == root and generation.parent == root:
        verify_doc_links(root)
    audit = verify(root, results, bench, generation)
    args.out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audit passed for ARC, OpenBookQA and all 48 generation cells; {audit['ledger']['total_seconds']} GPU s")


if __name__ == "__main__":
    main()
