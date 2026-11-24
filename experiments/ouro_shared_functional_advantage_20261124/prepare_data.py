"""Create experimental train folds without reusing the evaluation rows of the prior run."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from dataset_transfer import canonical
from prepare_pool import gsm, normalize_key, squad, valid
from task_data import hash_order, load, multiple_choice, save


SAMPLES = {"development": 96, "moment_holdout": 48, "moment_fit": 48, "fit": 840}


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def cases(ds, path):
    rows = load(path) if path.suffix == ".jsonl" else pq.read_table(path).to_pylist()
    if ds == "squad":
        return [r for row in rows if (r := squad(row))]
    if ds == "gsm8k":
        return [gsm(row, "process") for row in rows]
    return [multiple_choice(canonical(row, ds, ds, "train"), ds, "content") for row in rows]


def heldout(prior, sources):
    reserved = set()
    exposures = set()
    for file in (*prior.glob("dev_*.jsonl"), *prior.glob("test_*.jsonl")):
        reserved.update(normalize_key(x) for x in load(file))
    for file in (prior / "csqa_matched.jsonl", prior / "full_mixed_train.jsonl"):
        exposures.update(normalize_key(x) for x in load(file))
    for root in sources:
        for file in root.glob("*.jsonl"):
            if any(marker in file.stem for marker in ("confirm", "discovery", "validation")):
                reserved.update(normalize_key(x) for x in load(file))
    return reserved - {""}, exposures - {""}


def make_fold(tok, ordered, ds, group, wanted, used, exposures):
    selected = []
    for row in ordered:
        key = normalize_key(row)
        if not key or key in used or (group in ("development", "moment_holdout") and key in exposures):
            continue
        output = valid(row, tok, 576, 192)
        if output is None:
            continue
        if ds == "gsm8k" and (output["response_tokens"] < 12 or "####" not in output["response"]):
            continue
        selected.append(output)
        used.add(key)
        if len(selected) == wanted:
            return selected
    raise RuntimeError(f"Only {len(selected)}/{wanted} admissible {ds} rows for {group}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "arc", "csqa", "squad", "gsm8k", "prior", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--history", type=Path, action="append", default=[])
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=True)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "DATA_LOCK.json").exists():
        raise FileExistsError("The experimental split has already been locked")
    used, exposures = heldout(a.prior, a.history)
    info = {"files": {}, "source_sha256": {}, "length_limits": {"total_tokens": 576, "answer_tokens_including_eos": 192},
            "old_fit_rule": "development and moment_holdout exclude the two old checkpoints' actual training pools"}
    for ds, path in (("arc_challenge", a.arc), ("csqa", a.csqa), ("squad", a.squad), ("gsm8k", a.gsm8k)):
        source = cases(ds, path)
        info["source_sha256"][ds] = sha(path)
        remaining = {row["sample_id"] for row in source}
        if len(remaining) != len(source):
            raise ValueError(f"Source ID collision {ds}")
        for group, n in SAMPLES.items():
            ordered = sorted(source, key=lambda row: hash_order(row["sample_id"], f"functional-advantage/{ds}/{group}"))
            chosen = make_fold(tok, ordered, ds, group, n, used, exposures)
            file = a.out / f"{ds}_{group}.jsonl"
            save(file, chosen)
            info["files"][file.name] = dict(n=len(chosen), sha256=sha(file),
                                             ids=[row["sample_id"] for row in chosen],
                                             response_tokens=sum(row["response_tokens"] for row in chosen),
                                             overlap_old_fit=sum(normalize_key(row) in exposures for row in chosen),
                                             task_type_counts=dict(Counter(row["task_type"] for row in chosen)))
            print(ds, group, len(chosen), flush=True)
    (a.out / "DATA_LOCK.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
