"""Freeze development panels and full labeled confirmation inputs on the CPU."""

import argparse
from collections import Counter
import json
from pathlib import Path

from dataset_transfer import canonical
from prepare_external import Corpus, normalize_question, parquet, sha
from prepare_pool import gsm, squad
from task_data import hash_order, identity, load, multiple_choice, save


def choose(values, limit, salt):
    return sorted(values, key=lambda row: hash_order(row["sample_id"], salt))[:limit]


def choice_files(path, source):
    rows = parquet(path) if path.suffix == ".parquet" else load(path)
    result = []
    for row in rows:
        if "candidate" in row:
            raw = row["candidate"]
        else:
            raw = canonical(row, source, source, "official")
        result.append(multiple_choice(raw, source, "content"))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for a in ("data", "gen", "transfer", "advantage", "out"):
        p.add_argument(f"--{a}", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files, origin = {}, {}
    blocked = Corpus()
    for f in sorted(a.data.glob("*_train.jsonl")):
        for r in load(f):
            blocked.add(normalize_question(r), r["sample_id"])
    dev_arc = a.transfer / "inputs/upstream_arc_challenge_validation.parquet"
    dev_squad = a.gen / "inputs/squad_train.parquet"
    source_history = [a.gen / "data/full_mixed_train.jsonl", a.advantage / "data/squad_fit.jsonl",
                      a.advantage / "data/squad_development.jsonl", a.advantage / "data/squad_moment_holdout.jsonl"]
    protected_passage = {identity(r["context"]) for path in source_history for r in load(path) if r.get("dataset") == "squad"}
    protected_passage.update(identity(r["context"]) for r in parquet(a.gen / "inputs/squad_validation.parquet"))
    passages = {}
    excluded = Counter()
    for raw in parquet(dev_squad):
        passage = identity(raw["context"])
        if passage in protected_passage:
            excluded["historical_or_eval_passage"] += 1
            continue
        row = squad(raw)
        if row and blocked.match(normalize_question(row))[0] is None:
            current = passages.get(passage)
            if current is None or hash_order(row["sample_id"], "external/squad/article") < hash_order(current["sample_id"], "external/squad/article"):
                passages[passage] = row
    assert len(passages) >= 360
    dev = {"arc_challenge": choice_files(dev_arc, "arc_challenge"), "squad": list(passages.values())}
    for ds in ("csqa", "obqa", "gsm8k", "scienceqa"):
        historic = load(a.data / f"{ds}_development.jsonl")
        dev[ds] = ([item["views"]["process"] for item in historic] if ds == "gsm8k" else
                   [multiple_choice(item["candidate"], ds, "content") for item in historic])
    for ds, rows in dev.items():
        sorted_rows = choose(rows, 360, f"external/dev/{ds}")
        save(a.out / f"dev_{ds}.jsonl", sorted_rows)
    source = {"arc_challenge": a.transfer / "inputs/upstream_arc_challenge_test.parquet",
              "csqa": a.transfer / "data/csqa_validation.jsonl", "obqa": a.transfer / "inputs/obqa_test.parquet",
              "squad": a.gen / "inputs/squad_validation.parquet", "gsm8k": a.gen / "inputs/gsm8k_test.parquet"}
    for ds, raw in source.items():
        origin[ds] = dict(path=str(raw), sha256=sha(raw))
        if ds == "squad":
            rows = [squad(row) for row in parquet(raw)]
        elif ds == "gsm8k":
            rows = [gsm(row, "process") for row in parquet(raw)]
        else:
            rows = choice_files(raw, ds)
        for row in rows:
            if row is None or blocked.match(normalize_question(row))[0] is not None:
                raise ValueError(f"Selected training pool overlaps or lacks a test row: {ds}, {row and row['sample_id']}")
        save(a.out / f"test_{ds}.jsonl", rows)
    for path in sorted(a.out.glob("*.jsonl")):
        data = load(path)
        if len({row["sample_id"] for row in data}) != len(data):
            raise ValueError(f"Duplicate evaluation IDs in {path}")
        files[path.name] = dict(rows=len(data), sha256=sha(path))
    receipt = dict(files=files, sources=origin, dev_squad_history={str(path):sha(path) for path in source_history},
                   dev_squad_filtered=dict(excluded), dev_squad_article_count=len({identity(row["context"]) for row in dev["squad"]}),
                   dev_squad_policy="one question per normalized passage, no known old fitting, dev, moment holdout, or official validation passages",
                   dev_arc="historical official validation, excluded from joint training",
                   test_scope="historical full labeled panels, CSQA is author validation; development uses separate reserved cohorts")
    (a.out / "PANEL.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print({key: val["rows"] for key,val in files.items()}, flush=True)


if __name__ == "__main__":
    main()
