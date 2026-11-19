"""Verify released coverage and alignment independently of the analysis scripts."""
import argparse
from collections import Counter
import hashlib
import math
import json
from pathlib import Path


def jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def protocol_table(directory, phase, split, condition, methods):
    rows = list(jsonl(directory / f"{phase}.{split}.{condition}.jsonl"))
    n = {"train": 64, "validation": 299}[split]
    assert len(rows) == len({r["id"] for r in rows}) == n, (phase, split, condition)
    for row in rows:
        width = len(row["labels"])
        assert 0 <= row["gold"] < width
        assert sorted(row["order"]) == list(range(width))
        assert len(row["characters"]) == len(row["content_tokens"]) == width
        assert len(row["continuation_tokens"]) == width
        assert row["context_tokens"] > 0
        if condition in "EF":
            assert row["continuation_tokens"] == [1] * width
        assert set(row["scores"]) == methods, (phase, row["id"], row["scores"].keys())
        for scores in row["scores"].values():
            assert len(scores) == width and all(math.isfinite(v) for v in scores)
    return rows


def verify_protocol(directory):
    native = {f"native_k{k}" for k in range(1, 5)}
    loopcd = {f"loopcd_{tag}_k{k}" for tag in ("fixed", "adaptive", "hidden") for k in range(2, 5)}
    ids = {}
    for split in ("train", "validation"):
        anchor = None
        contexts = {}
        for condition in "ABCDEF":
            a = protocol_table(directory, "native", split, condition, native)
            b = protocol_table(directory, "loopcd", split, condition, loopcd)
            ordering = [(r["id"], r["gold"], r["labels"], r["characters"]) for r in a]
            if anchor is None:
                anchor = ordering
            assert ordering == anchor, (split, condition, "item/choice mismatch")
            contexts[condition] = [r["context_ids_sha256"] for r in a]
            for initial, guided in zip(a, b, strict=True):
                for key in ("id", "gold", "labels", "order", "prompt_sha256", "context_ids_sha256",
                            "characters", "content_tokens", "continuation_tokens", "context_tokens"):
                    assert initial[key] == guided[key], (split, condition, initial["id"], key)
        assert contexts["C"] == contexts["E"], (split, "zero-shot content/letter context mismatch")
        ids[split] = {case[0] for case in anchor}
    assert ids["train"].isdisjoint(ids["validation"])
    return {"protocols": 6, "phases": 2, "train": len(ids["train"]), "validation": len(ids["validation"]),
            "zero_shot_content_and_letter_contexts_equal": True}


def verify_atlas(atlas, real, hard=None):
    completion = json.loads((atlas / "tableaux_complete.json").read_text())
    native = json.loads((atlas / "native_gate.json").read_text())
    baseline_forks = {}
    for row in jsonl(atlas / "natural.jsonl"):
        if row["fork"] not in ("open", "content", "letter"):
            continue
        key = (row["id"], row["fork"])
        worlds = baseline_forks.setdefault(key, set())
        world = (row["s"], row["m"])
        assert world not in worlds, (key, world)
        worlds.add(world)
        assert len(row["scores"]) == 4 and all(len(k) == 2 for k in row["scores"])
    all_worlds = {(s, m) for s in range(3) for m in range(3)}
    assert all(worlds == all_worlds for worlds in baseline_forks.values())
    group_counts = Counter(gid.rsplit("-", 1)[0] for gid, fork in baseline_forks if fork == "open")
    assert group_counts == {"development": 24, "new_entities": 48, "new_structure": 48}, group_counts
    assert len(baseline_forks) == completion["groups"] * 3 == 360
    assert native["independent_native_wrapper_bitwise"] and native["depths"] == 4
    parity = json.loads((atlas / "prefix_parity.json").read_text())
    assert {r["id"] for r in parity} == {gid for gid, _ in baseline_forks}
    for row in parity:
        assert row["before_divergence_tokens"] > 0
        assert row["all_tensor_sites"] == 196
        assert row["exact_tensor_sites_by_fork"] == [196] * len(row["forks"])
    real_info = json.loads((real / "hotpot_complete.json").read_text())
    assert real_info["groups"] == 64 and 64 <= real_info["eligible"] <= real_info["original_split_rows"] == 7405
    result = {"factorial_groups": completion["groups"], "groups_by_split": dict(group_counts),
              "worlds_per_group": 9, "hotpot_groups": real_info["groups"],
              "prefix_sites_per_branch": 196}
    if hard is not None or (real / "hotpot_hard_complete.json").exists():
        hard_info = json.loads(((hard or real) / "hotpot_hard_complete.json").read_text())
        assert hard_info["groups"] == 48
        result["hotpot_disjoint_hard_groups"] = hard_info["groups"]
    return result


def verify_generation(directory):
    config = json.loads((directory / "GENERATION_COST_CONFIG.json").read_text())
    metrics = json.loads((directory / "GENERATION_COST_METRICS.json").read_text())
    timed = config["timed_ids"]
    assert len(timed) == len(set(timed)) == 16 and config["warmup_id"] not in timed
    assert set(metrics) == set("ABCDEF")
    native = {f"native_k{k}" for k in range(1, 5)}
    loopcd = {f"loopcd_{tag}_k{k}" for tag in ("fixed", "adaptive", "hidden") for k in range(2, 5)}
    for condition, entries in metrics.items():
        assert set(entries) == native | loopcd
        for method, stats in entries.items():
            path = directory / f"generation.{condition}.{method}.json"
            data = path.read_bytes()
            records = json.loads(data)
            assert stats["records_sha256"] == hashlib.sha256(data).hexdigest()
            assert stats["rows"] == len(records) == len(timed)
            assert [r["id"] for r in records] == timed and stats["exact_correct"] == sum(
                r["exact"] == r["gold"] for r in records)
            for row in records:
                sample = [row["candidate_seconds"], row["prefill_seconds"], *row["decode_seconds"]]
                assert all(math.isfinite(t) and t > 0 for t in sample)
                assert len(row["generated_ids"]) <= config["budget_new_tokens"]
    return {"timed_questions": len(timed), "protocol_method_cells": sum(map(len, metrics.values()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("hotpot", type=Path)
    parser.add_argument("--hotpot-hard", type=Path)
    args = parser.parse_args()
    report = {"protocol": verify_protocol(args.protocol),
              "atlas": verify_atlas(args.atlas, args.hotpot, args.hotpot_hard),
              "generation": verify_generation(args.protocol)}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
