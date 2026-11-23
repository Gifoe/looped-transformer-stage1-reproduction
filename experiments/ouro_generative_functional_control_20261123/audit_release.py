"""Verify complete frozen inference receipts and emit a text-free data manifest."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path

from task_data import hash_order, load


def sha(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(test, results, allow_shortfall=False):
    config = read(test)
    fingerprint = sha(test)
    if sha(Path(config["controller_file"])) != config["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint fingerprint mismatch")
    expected = set()
    missing = []
    for case in config["cases"]:
        data = Path(case["file"])
        if sha(data) != case["data_sha256"]:
            raise RuntimeError(f"Changed evaluation data {data}")
        dataset = case["dataset"]
        ordered = sorted(load(data), key=lambda r: hash_order(r["sample_id"], f"test/{dataset}"))
        for depth in case["depths"]:
            count = min(len(ordered), case["full_limit"] if depth == 4 else case["transfer_limit"])
            ids = [r["sample_id"] for r in ordered[:count]]
            if len(set(ids)) != count:
                raise RuntimeError(f"Non-unique IDs in {dataset}")
            for arm in config["allowed_run_order"]:
                key = f"{dataset}__{arm}__k{depth}__{case['mode']}"
                expected.add(key)
                path = results / (key + ".jsonl")
                if not path.with_suffix(".json").exists():
                    if not allow_shortfall:
                        raise FileNotFoundError(path.with_suffix(".json"))
                    missing.append(key)
                    continue
                receipt = read(path.with_suffix(".json"))
                plan = receipt["plan"]
                signatures = {
                    "protocol_sha256": fingerprint,
                    "data_sha256": case["data_sha256"],
                    "checkpoint_sha256": config["arms"][arm]["sha256"],
                    "n": count,
                    "batch": case["batch"],
                    "max_new": case["max_new"],
                }
                if any(plan[k] != v for k, v in signatures.items()) or receipt["rows_sha256"] != sha(path):
                    raise RuntimeError(f"Mismatched fingerprint or plan {key}")
                if [row["id"] for row in load(path)] != ids:
                    raise RuntimeError(f"Mismatched test population {key}")
    for extension in ("json", "jsonl"):
        existing = {path.stem for path in results.glob(f"*.{extension}")}
        if existing - expected or expected - existing - set(missing):
            raise RuntimeError(f"Unexpected {extension} files")
    return dict(shards=len(expected)-len(missing), planned_shards=len(expected),
                protocol_sha256=fingerprint, selected_arm=config["selected_arm"], missing=missing)


def public_index(data, target):
    target.mkdir(parents=True, exist_ok=True)
    files = [*data.glob("*.jsonl"), *(data / "filtered").glob("*.jsonl")]
    manifest = {}
    for file in sorted(files):
        if not file.name.startswith(("dev_", "test_")) and file.name not in {
            "csqa_matched.jsonl", "mixed_train.jsonl", "full_mixed_train.jsonl", "retention_train.jsonl"
        }:
            continue
        path = str(file.relative_to(data))
        rows = load(file)
        identifiers = Counter(row["sample_id"] for row in rows)
        collisions = [
            dict(row_index=index, id=row["sample_id"], prompt_sha256=sha256(row["prompt"].encode()).hexdigest())
            for index, row in enumerate(rows) if identifiers[row["sample_id"]] > 1
        ]
        manifest[path] = dict(
            sha256=sha(file),
            n=len(rows),
            distinct_sample_ids=len(identifiers),
            source_id_collisions=collisions,
            by_dataset=dict(Counter(row["dataset"] for row in rows)),
            task_types=dict(Counter(row["task_type"] for row in rows)),
            ordered_ids=[row["sample_id"] for row in rows],
        )
    (target / "DATA_INDEX.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--verify-test", action="store_true")
    parser.add_argument("--allow-budget-shortfall", action="store_true")
    args = parser.parse_args()
    if args.index is not None:
        public_index(Path("data"), args.index)
    if args.verify_test:
        evidence = verify(Path("results/FINAL_PROTOCOL.json"), Path("results/confirmation"), args.allow_budget_shortfall)
        Path("results/CONFIRMATION_COVERAGE.json").write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
