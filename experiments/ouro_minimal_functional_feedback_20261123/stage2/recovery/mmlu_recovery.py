"""Export MMLU via the original pinned Harness tasks and canonical subject shards."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import quote

import datasets
import pyarrow as pa
import pyarrow.parquet as parquet
import requests

import stage2_requests as exporter


REPO = "cais/mmlu"
REVISION = exporter.DATASET_REVISIONS[REPO]
MIRROR = "https://hf-mirror.com"
SHARD = re.compile(r"([a-z0-9_]+)/(dev|test)-00000-of-00001\.parquet")


def fetch(src):
    for attempt in range(3):
        try:
            response = requests.get(src, timeout=(8, 18))
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def file_receipt(entry, root, hub_blobs):
    name = entry["rfilename"]
    signature = entry["lfs"]["sha256"]
    path = root / name
    if path.exists():
        body = path.read_bytes()
        source = "recovery_cache"
    elif (hub_blobs / signature).is_file():
        body = (hub_blobs / signature).read_bytes()
        source = "original_verified_hub_blob"
    else:
        body = fetch(f"{MIRROR}/datasets/{REPO}/resolve/{REVISION}/{quote(name, safe='/')}")
        source = "pinned_mirror"
    if hashlib.sha256(body).hexdigest() != signature or len(body) != entry["size"]:
        raise RuntimeError(f"pinned shard integrity mismatch: {name}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    subject, split = SHARD.fullmatch(name).groups()
    return subject, split, {
        "file": name,
        "sha256": signature,
        "bytes": len(body),
        "rows": parquet.ParquetFile(path).metadata.num_rows,
        "source": source,
    }


def verify_original_arrow(source, canonical, subject, split):
    candidates = sorted(source.glob(f"{subject}/0.0.0/*/mmlu-{split}.arrow"))
    if not candidates:
        return None
    for path in candidates:
        with pa.memory_map(str(path), "r") as stream:
            original = pa.ipc.open_stream(stream).read_all()
        if not canonical.equals(original, check_metadata=False):
            raise RuntimeError(f"original Arrow data or order differs for {subject}/{split}")
    return {"files": len(candidates), "equal_rows_in_order": canonical.num_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.workdir
    root.mkdir(parents=True, exist_ok=True)
    meta = json.loads(fetch(f"{MIRROR}/api/datasets/{REPO}/revision/{REVISION}?blobs=true"))
    if meta["sha"] != REVISION:
        raise RuntimeError("metadata returned another MMLU commit")
    shards = sorted(
        (
            item for item in meta["siblings"]
            if SHARD.fullmatch(item["rfilename"])
            and not item["rfilename"].startswith(("all/", "auxiliary_train/"))
        ),
        key=lambda item: item["rfilename"],
    )
    if len(shards) != 114:
        raise RuntimeError(f"expected 114 canonical shards, got {len(shards)}")
    print(f"Fetching canonical pinned MMLU shards ({sum(x['size'] for x in shards)} bytes)", flush=True)
    hub_blobs = args.source_cache / "hub/datasets--cais--mmlu/blobs"
    source_arrow = args.source_cache / "datasets/cais___mmlu"
    records = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for subject, split, receipt in pool.map(lambda item: file_receipt(item, root, hub_blobs), shards):
            records[subject][split] = receipt
    if len(records) != 57 or any(set(splits) != {"test", "dev"} for splits in records.values()):
        raise RuntimeError("incomplete subject/split coverage")
    split_sizes = {
        split: sum(value[split]["rows"] for value in records.values())
        for split in ("dev", "test")
    }
    if split_sizes != {"dev": 285, "test": 14042}:
        raise RuntimeError(f"unexpected canonical MMLU cardinality {split_sizes}")
    for subject, splits in records.items():
        if splits["dev"]["rows"] != 5:
            raise RuntimeError(f"{subject}: expected exactly five dev rows")
        for split, receipt in splits.items():
            table = parquet.read_table(root / receipt["file"])
            receipt["original_arrow_parity"] = verify_original_arrow(source_arrow, table, subject, split)
    print(f"Pinned canonical split coverage {split_sizes}", flush=True)
    (root / "canonical_mmlu_receipt.json").write_text(
        json.dumps(
            {"dataset": REPO, "revision": REVISION, "split_sizes": split_sizes, "subjects": records},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    # Change only data transport. The original Harness task configs, subject
    # description, sampler, renderer and Instance construction still run.
    loaded = set()

    def pinned_subject_dataset(path, name=None, **kwargs):
        if path != REPO or kwargs.get("revision") != REVISION or name not in records:
            raise RuntimeError(f"unexpected data load: {path}:{name}")
        loaded.add(name)
        return datasets.DatasetDict({
            split: datasets.Dataset(parquet.read_table(root / receipt["file"]))
            for split, receipt in records[name].items()
        })

    from unittest.mock import patch
    with patch.object(datasets, "load_dataset", new=pinned_subject_dataset):
        sys.argv = [
            str(Path(exporter.__file__).name), "--tasks", "mmlu", "--output",
            str(root / "mmlu.jsonl"),
        ]
        exporter.main()
    if loaded != records.keys():
        raise RuntimeError("Harness subjects and canonical Hub subjects differ")


if __name__ == "__main__":
    main()
