"""Validate and join separately measured MMLU-Pro LoopCD arms."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("auxiliary", type=Path)
    args = parser.parse_args()
    full = args.primary
    extra = args.auxiliary
    lock = digest(full / "FREEZE.json")
    if digest(extra / "FREEZE.json") != lock:
        raise ValueError("The two runs did not use the same frozen LoopCD evaluation")
    results = read(full / "RESULTS.json")
    additional = read(extra / "RESULTS.json")
    row = results["mmlu_pro"]
    if (set(additional) != {"mmlu_pro"} or set(row["arms"]) not in ({"fixed"}, {"fixed", "adaptive"})
            or set(additional["mmlu_pro"]["arms"]) != {"adaptive"}):
        raise ValueError("Unexpected MMLU-Pro coverage on the two workers")
    incoming = additional["mmlu_pro"]
    if any(row[key] != incoming[key] for key in ("source", "joint_candidate_time", "primary")):
        raise ValueError("The MMLU-Pro results use different protocols")
    item = incoming["arms"]["adaptive"]
    receipt = extra / "mmlu_pro__adaptive.json"
    predictions = receipt.with_suffix(".jsonl")
    expected = item["rows_sha256"]
    stamp = read(receipt)
    if (stamp["rows_sha256"] != expected or digest(predictions) != expected
            or stamp["plan"]["protocol_sha256"] != lock
            or item["metrics"][row["primary"]]["versus_native"]["n"] != 12032):
        raise ValueError("The adaptive generation predictions do not match their full receipt")
    with predictions.open(encoding="utf-8") as stream:
        predicted_ids = {json.loads(line)["id"] for line in stream}
    with (full / "mmlu_pro__fixed.jsonl").open(encoding="utf-8") as stream:
        fixed_ids = {json.loads(line)["id"] for line in stream}
    if len(predicted_ids) != 12032 or predicted_ids != fixed_ids:
        raise ValueError("The two generation arms did not score the same questions")
    if "adaptive" in row["arms"] and row["arms"]["adaptive"] != item:
        raise ValueError("A different adaptive run already exists")
    for artifact in (predictions, receipt):
        target = full / artifact.name
        staging = target.with_suffix(target.suffix + ".partial")
        shutil.copyfile(artifact, staging)
        staging.replace(target)
    row["arms"]["adaptive"] = item
    path = full / "RESULTS.json"
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print("Verified 12,032 matched adaptive predictions", expected, flush=True)


if __name__ == "__main__":
    main()
